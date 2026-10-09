# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import asyncio
import inspect
import math
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from numbers import Real
from statistics import mean
from typing import Generic, TypeVar

InputT = TypeVar("InputT")
OutputT = TypeVar("OutputT")
Scorer = Callable[[OutputT, OutputT], float | Awaitable[float]]


def _finite_score(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError("Scores must be real numbers, not booleans")
    try:
        result = float(value)
    except (OverflowError, ValueError) as error:
        raise ValueError("Scores must be finite") from error
    if not math.isfinite(result):
        raise ValueError("Scores must be finite")
    return result


def _validate_concurrency(concurrency: int) -> None:
    if isinstance(concurrency, bool) or not isinstance(concurrency, int):
        raise TypeError("concurrency must be a positive integer")
    if concurrency <= 0:
        raise ValueError("concurrency must be a positive integer")


@dataclass(frozen=True, slots=True)
class EvaluationExample(Generic[InputT, OutputT]):
    input: InputT
    expected: OutputT


@dataclass(frozen=True, slots=True)
class EvaluationResult(Generic[OutputT]):
    expected: OutputT
    actual: OutputT
    score: float
    metrics: dict[str, float] = field(default_factory=dict, kw_only=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "score", _finite_score(self.score))
        object.__setattr__(
            self,
            "metrics",
            {name: _finite_score(value) for name, value in self.metrics.items()},
        )


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    count: int
    mean_score: float
    metrics: dict[str, float]


def summarize(results: Sequence[EvaluationResult[OutputT]]) -> EvaluationSummary:
    """Return macro means; empty results have score 0 and no metric means.

    A metric present on only some results is averaged over those results.
    """
    values: dict[str, list[float]] = {}
    for result in results:
        for name, value in result.metrics.items():
            values.setdefault(name, []).append(_finite_score(value))
    return EvaluationSummary(
        count=len(results),
        mean_score=(
            mean(_finite_score(result.score) for result in results) if results else 0.0
        ),
        metrics={name: mean(scores) for name, scores in values.items()},
    )


async def _score(scorer: Scorer[OutputT], expected: OutputT, actual: OutputT) -> float:
    value = scorer(expected, actual)
    if inspect.isawaitable(value):
        value = await value
    return _finite_score(value)


class DatasetRunner(Generic[InputT, OutputT]):
    def __init__(
        self,
        predict: Callable[[InputT], Awaitable[OutputT]],
        score: Scorer[OutputT],
        *,
        metrics: Mapping[str, Scorer[OutputT]] | None = None,
    ) -> None:
        self.predict, self.score = predict, score
        self.metrics = dict(metrics) if metrics is not None else {}

    async def run(
        self,
        examples: Sequence[EvaluationExample[InputT, OutputT]],
        *,
        concurrency: int = 8,
    ) -> list[EvaluationResult[OutputT]]:
        """Evaluate in input order with at most concurrency live workers.

        Predictor/scorer failures and caller cancellation cancel and await workers.
        """
        _validate_concurrency(concurrency)
        dataset = tuple(examples)
        pending = iter(enumerate(dataset))
        results: dict[int, EvaluationResult[OutputT]] = {}

        async def worker() -> None:
            # Claim an index without awaiting, so workers cannot claim it twice.
            for index, example in pending:
                actual = await self.predict(example.input)
                score = await _score(self.score, example.expected, actual)
                metrics = {
                    name: await _score(scorer, example.expected, actual)
                    for name, scorer in self.metrics.items()
                }
                results[index] = EvaluationResult(
                    example.expected, actual, score, metrics=metrics
                )

        tasks = [
            asyncio.create_task(worker()) for _ in range(min(concurrency, len(dataset)))
        ]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            # Cleanup must also run for CancelledError, not only Exception.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        return [results[index] for index in range(len(dataset))]


def accuracy(expected: str, actual: str) -> float:
    """Case- and whitespace-sensitive exact match, including two empty strings."""
    return float(expected == actual)


def exact_match(expected: str, actual: str) -> float:
    return accuracy(expected, actual)


def _token_counts(expected: str, actual: str) -> tuple[int, int, int]:
    """Whitespace tokens, case-sensitive, with duplicate multiplicity preserved."""
    expected_tokens = Counter(expected.split())
    actual_tokens = Counter(actual.split())
    overlap = sum((expected_tokens & actual_tokens).values())
    return overlap, sum(expected_tokens.values()), sum(actual_tokens.values())


def token_precision(expected: str, actual: str) -> float:
    """Both token lists empty scores 1; exactly one empty scores 0."""
    overlap, expected_count, actual_count = _token_counts(expected, actual)
    if not expected_count or not actual_count:
        return float(expected_count == actual_count)
    return overlap / actual_count


def token_recall(expected: str, actual: str) -> float:
    """Both token lists empty scores 1; exactly one empty scores 0."""
    overlap, expected_count, actual_count = _token_counts(expected, actual)
    if not expected_count or not actual_count:
        return float(expected_count == actual_count)
    return overlap / expected_count


def token_f1(expected: str, actual: str) -> float:
    """Harmonic mean of token precision and recall; two empty lists score 1."""
    overlap, expected_count, actual_count = _token_counts(expected, actual)
    total = expected_count + actual_count
    return 2.0 * overlap / total if total else 1.0


class PromptRunner:
    """Compare named format templates using an injected async string predictor.

    Inputs are explicit mappings for template.format_map(input). Expected answers
    are used only for scoring, never implicitly added to template data. Prompts
    run sequentially, each with its own bounded DatasetRunner worker pool.
    """

    def __init__(
        self,
        predict: Callable[[str], Awaitable[str]],
        prompts: Mapping[str, str],
        *,
        score: Scorer[str] = accuracy,
        metrics: Mapping[str, Scorer[str]] | None = None,
    ) -> None:
        self.predict = predict
        self.prompts = dict(prompts)
        self.score = score
        self.metrics = (
            dict(metrics)
            if metrics is not None
            else {
                "exact_match": exact_match,
                "token_precision": token_precision,
                "token_recall": token_recall,
                "token_f1": token_f1,
            }
        )

    async def run(
        self,
        examples: Sequence[EvaluationExample[Mapping[str, object], str]],
        *,
        concurrency: int = 8,
    ) -> dict[str, list[EvaluationResult[str]]]:
        _validate_concurrency(concurrency)
        dataset = tuple(
            EvaluationExample(dict(example.input), example.expected)
            for example in examples
        )
        results: dict[str, list[EvaluationResult[str]]] = {}
        for name, template in self.prompts.items():
            # Materialize formatting before starting this prompt's predictions.
            formatted = [
                EvaluationExample(template.format_map(example.input), example.expected)
                for example in dataset
            ]
            runner = DatasetRunner(self.predict, self.score, metrics=self.metrics)
            results[name] = await runner.run(formatted, concurrency=concurrency)
        return results
