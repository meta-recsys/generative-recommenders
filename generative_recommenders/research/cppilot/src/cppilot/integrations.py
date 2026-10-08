# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import asyncio
import hmac
import importlib
import math
import re
import sqlite3
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, cast


class AuthProvider(ABC):
    @abstractmethod
    async def authenticate(self, credential: str | None) -> str: ...


class JobBackend(ABC):
    @abstractmethod
    async def submit(self, name: str, payload: Mapping[str, Any]) -> str: ...

    @abstractmethod
    async def status(self, job_id: str) -> Mapping[str, Any]: ...


class MetricsBackend(ABC):
    @abstractmethod
    async def increment(
        self, name: str, value: float = 1, *, labels: Mapping[str, str] | None = None
    ) -> None: ...


class SQLQueryBackend(ABC):
    """Execute bound, read-only queries; implementations must enforce this in the DB.

    SQL text filtering is not a security boundary. Remote implementations should
    use a read-only database role/transaction and disable side-effecting functions.
    """

    @abstractmethod
    async def query(
        self, statement: str, parameters: Sequence[Any] = ()
    ) -> list[Mapping[str, Any]]: ...


class ObjectStore(ABC):
    @abstractmethod
    async def read(self, uri: str) -> bytes: ...

    @abstractmethod
    async def write(self, uri: str, value: bytes) -> None: ...

    @abstractmethod
    def list(self, uri: str) -> AsyncIterator[str]:
        """Return an async iterator directly, without awaiting the method."""
        ...


class BearerTokenAuth(AuthProvider):
    """Authenticate one configured bearer token, returning its fixed principal."""

    def __init__(self, token: str, *, principal: str = "authenticated") -> None:
        if not token or any(character.isspace() for character in token):
            raise ValueError("token must be nonempty and contain no whitespace")
        if not principal:
            raise ValueError("principal must be nonempty")
        self._token = token.encode("utf-8")
        self._principal = principal

    async def authenticate(self, credential: str | None) -> str:
        parts = credential.split() if credential is not None else []
        well_formed = len(parts) == 2 and parts[0].lower() == "bearer"
        candidate = parts[1].encode("utf-8") if well_formed else b""
        matches = hmac.compare_digest(candidate, self._token)
        if not well_formed or not matches:
            raise PermissionError("invalid bearer credential")
        return self._principal


class SQLiteReadOnlyBackend(SQLQueryBackend):
    """Query an existing SQLite file with a fresh, restricted connection per call.

    Results are truncated to max_rows. Only allowlisted built-in functions may
    run; all caller PRAGMAs, writes, transactions and ATTACH/DETACH are denied.
    The deadline bounds SQLite VM execution, not OS I/O or individual functions.
    A database file must be trusted as a file format, not just as SQL content.
    """

    _FUNCTIONS = frozenset(
        "abs avg char coalesce concat concat_ws count format glob hex ifnull iif "
        "instr length like likelihood likely lower ltrim max min nullif octet_length "
        "printf quote replace round rtrim sign substr substring sum total trim "
        "typeof unicode unlikely upper zeroblob date datetime julianday strftime "
        "time timediff unixepoch group_concat string_agg row_number rank dense_rank "
        "percent_rank cume_dist ntile lag lead first_value last_value nth_value "
        "json json_array json_array_length json_error_position json_extract "
        "json_group_array json_group_object json_insert json_object json_patch "
        "json_quote json_remove json_replace json_set json_type json_valid "
        "jsonb jsonb_array jsonb_extract jsonb_group_array jsonb_group_object "
        "jsonb_insert jsonb_object jsonb_patch jsonb_remove jsonb_replace jsonb_set".split()
    )

    def __init__(
        self,
        path: str | Path,
        *,
        max_rows: int = 1000,
        timeout: float = 5.0,
    ) -> None:
        if isinstance(max_rows, bool) or not isinstance(max_rows, int) or max_rows < 1:
            raise ValueError("max_rows must be a positive integer")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        database = Path(path).resolve(strict=True)
        if not database.is_file():
            raise ValueError("path must identify an existing database file")
        self._uri = database.as_uri() + "?mode=ro"
        self._max_rows = max_rows
        self._timeout = timeout

    @classmethod
    def _authorize(
        cls,
        action: int,
        argument1: str | None,
        argument2: str | None,
        database: str | None,
        source: str | None,
    ) -> int:
        if action in (
            sqlite3.SQLITE_SELECT,
            sqlite3.SQLITE_READ,
            sqlite3.SQLITE_RECURSIVE,
        ):
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_FUNCTION:
            function = (argument2 or "").lower()
            if function in cls._FUNCTIONS:
                return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    async def query(
        self, statement: str, parameters: Sequence[Any] = ()
    ) -> list[Mapping[str, Any]]:
        return await asyncio.to_thread(self._query, statement, tuple(parameters))

    def _query(
        self, statement: str, parameters: tuple[Any, ...]
    ) -> list[Mapping[str, Any]]:
        connection = sqlite3.connect(self._uri, uri=True, timeout=self._timeout)
        try:
            connection.execute("PRAGMA query_only = ON")
            connection.execute("PRAGMA trusted_schema = OFF")
            connection.set_authorizer(self._authorize)
            deadline = time.monotonic() + self._timeout
            connection.set_progress_handler(
                lambda: int(time.monotonic() >= deadline), 1000
            )
            connection.row_factory = sqlite3.Row
            cursor = connection.execute(statement, parameters)
            return [dict(row) for row in cursor.fetchmany(self._max_rows)]
        finally:
            connection.close()


class FsspecObjectStore(ObjectStore):
    """Use fsspec URLs, or an injected synchronous fsspec filesystem.

    Listing is a nonrecursive directory listing, including directory entries.
    fsspec's ls materializes the listing before it can be yielded; this is not
    paginated streaming. Writes inherit the filesystem's overwrite semantics.
    Protocol-specific packages and credentials are the caller's responsibility.
    """

    def __init__(
        self,
        *,
        filesystem: Any = None,
        storage_options: Mapping[str, Any] | None = None,
    ) -> None:
        self._filesystem = filesystem
        self._storage_options = dict(storage_options or {})

    def _resolve(self, uri: str) -> tuple[Any, str]:
        if self._filesystem is not None:
            return self._filesystem, self._filesystem._strip_protocol(uri)
        fsspec = importlib.import_module("fsspec.core")
        return cast(tuple[Any, str], fsspec.url_to_fs(uri, **self._storage_options))

    def _read(self, uri: str) -> bytes:
        filesystem, path = self._resolve(uri)
        with filesystem.open(path, "rb") as stream:
            return cast(bytes, stream.read())

    async def read(self, uri: str) -> bytes:
        return await asyncio.to_thread(self._read, uri)

    def _write(self, uri: str, value: bytes) -> None:
        filesystem, path = self._resolve(uri)
        with filesystem.open(path, "wb") as stream:
            stream.write(value)

    async def write(self, uri: str, value: bytes) -> None:
        await asyncio.to_thread(self._write, uri, value)

    def _list(self, uri: str) -> list[str]:
        filesystem, path = self._resolve(uri)
        return [
            filesystem.unstrip_protocol(entry)
            for entry in filesystem.ls(path, detail=False)
        ]

    async def list(self, uri: str) -> AsyncIterator[str]:
        for entry in await asyncio.to_thread(self._list, uri):
            yield entry


class KubernetesJobBackend(JobBackend):
    """Submit structured batch/v1 Job manifests through the synchronous SDK.

    payload is a complete Job manifest. name and the explicit namespace override
    metadata. Commands/args are passed unchanged; no shell commands are generated.
    This is not a manifest security sandbox: apply cluster RBAC/admission policy.
    Inject a BatchV1Api, or select in-cluster/default kubeconfig loading explicitly.
    """

    def __init__(
        self,
        namespace: str,
        *,
        api: Any = None,
        in_cluster: bool = True,
        kubeconfig: str | None = None,
    ) -> None:
        self._validate_name(namespace)
        if in_cluster and kubeconfig is not None:
            raise ValueError("kubeconfig requires in_cluster=False")
        self._namespace = namespace
        self._api = api
        self._in_cluster = in_cluster
        self._kubeconfig = kubeconfig
        self._lock = threading.Lock()

    @staticmethod
    def _validate_name(name: str) -> None:
        if (
            len(name) > 63
            or re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", name) is None
        ):
            raise ValueError("name must be a DNS label of at most 63 characters")

    def _get_api(self) -> Any:
        with self._lock:
            if self._api is None:
                client = importlib.import_module("kubernetes.client")
                config = importlib.import_module("kubernetes.config")
                configuration = client.Configuration()
                if self._in_cluster:
                    config.load_incluster_config(client_configuration=configuration)
                else:
                    config.load_kube_config(
                        config_file=self._kubeconfig, client_configuration=configuration
                    )
                self._api = client.BatchV1Api(client.ApiClient(configuration))
            return self._api

    async def submit(self, name: str, payload: Mapping[str, Any]) -> str:
        self._validate_name(name)
        manifest = deepcopy(dict(payload))
        if manifest.get("apiVersion") != "batch/v1" or manifest.get("kind") != "Job":
            raise ValueError("payload must be a batch/v1 Job manifest")
        if not isinstance(manifest.get("spec"), Mapping):
            raise ValueError("Job manifest requires a spec mapping")
        metadata = dict(manifest.get("metadata", {}))
        metadata.pop("generateName", None)
        metadata["name"] = name
        metadata["namespace"] = self._namespace
        manifest["metadata"] = metadata
        await asyncio.to_thread(self._submit, manifest)
        return name

    def _submit(self, manifest: Mapping[str, Any]) -> None:
        self._get_api().create_namespaced_job(namespace=self._namespace, body=manifest)

    async def status(self, job_id: str) -> Mapping[str, Any]:
        self._validate_name(job_id)
        return await asyncio.to_thread(self._status, job_id)

    def _status(self, job_id: str) -> Mapping[str, Any]:
        job = self._get_api().read_namespaced_job_status(
            name=job_id, namespace=self._namespace
        )
        return dict(job) if isinstance(job, Mapping) else job.to_dict()


def _counter_value(value: float) -> float:
    if not math.isfinite(value) or value < 0:
        raise ValueError("counter increments must be finite and nonnegative")
    return float(value)


class PrometheusMetricsBackend(MetricsBackend):
    """Lazily register counters in an injected or adapter-private registry.

    Label names are fixed on the first increment for each metric; later calls
    must use exactly that set. Values may vary. No HTTP server is started.
    Share this adapter, not separate adapters, for counters in a shared registry.
    """

    def __init__(self, *, sdk: Any = None, registry: Any = None) -> None:
        self._sdk = sdk
        self._registry = registry
        self._counters: dict[str, tuple[tuple[str, ...], Any]] = {}
        self._lock = threading.Lock()

    @property
    def registry(self) -> Any:
        """The supplied registry, or None until a private one is first used."""
        return self._registry

    async def increment(
        self, name: str, value: float = 1, *, labels: Mapping[str, str] | None = None
    ) -> None:
        amount = _counter_value(value)
        await asyncio.to_thread(self._increment, name, amount, dict(labels or {}))

    def _increment(self, name: str, value: float, labels: dict[str, str]) -> None:
        label_names = tuple(sorted(labels))
        with self._lock:
            entry = self._counters.get(name)
            if entry is None:
                if self._sdk is None:
                    self._sdk = importlib.import_module("prometheus_client")
                if self._registry is None:
                    self._registry = self._sdk.CollectorRegistry()
                counter = self._sdk.Counter(
                    name,
                    f"CPPilot counter: {name}",
                    labelnames=label_names,
                    registry=self._registry,
                )
                self._counters[name] = (label_names, counter)
            else:
                expected, counter = entry
                if label_names != expected:
                    raise ValueError(f"label names changed for counter {name!r}")
            if label_names:
                counter = counter.labels(**labels)
            counter.inc(value)


class MLflowMetricsBackend(MetricsBackend):
    """Log cumulative counter values to an explicit, already-existing MLflow run.

    Labels are unsupported and rejected instead of mutating global run tags.
    Totals/steps are local to this adapter and reset when it is recreated. A
    transport failure is ambiguous: retries may repeat a point on the server.
    No run is created, activated, ended, or selected through MLflow global state.
    """

    def __init__(
        self,
        run_id: str,
        *,
        client: Any = None,
        tracking_uri: str | None = None,
    ) -> None:
        if not run_id:
            raise ValueError("run_id must be explicit and nonempty")
        self._run_id = run_id
        self._client = client
        self._tracking_uri = tracking_uri
        self._totals: dict[str, float] = {}
        self._steps: dict[str, int] = {}
        self._lock = threading.Lock()

    async def increment(
        self, name: str, value: float = 1, *, labels: Mapping[str, str] | None = None
    ) -> None:
        if labels:
            raise ValueError("MLflow counter labels are not supported")
        amount = _counter_value(value)
        await asyncio.to_thread(self._increment, name, amount)

    def _increment(self, name: str, value: float) -> None:
        with self._lock:
            if self._client is None:
                tracking = importlib.import_module("mlflow.tracking")
                self._client = tracking.MlflowClient(tracking_uri=self._tracking_uri)
            total = self._totals.get(name, 0.0) + value
            if not math.isfinite(total):
                raise ValueError("cumulative counter overflow")
            step = self._steps.get(name, 0)
            self._client.log_metric(
                run_id=self._run_id,
                key=name,
                value=total,
                timestamp=int(time.time() * 1000),
                step=step,
            )
            self._totals[name] = total
            self._steps[name] = step + 1
