# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-unsafe

import logging
import os
import random
import time
from datetime import date
from typing import Dict, Optional

import gin
import torch
import torch.distributed as dist
from generative_recommenders.research.data.eval import (
    _avg,
    add_to_summary_writer,
    eval_metrics_v2_from_tensors,
    get_eval_state,
)
from generative_recommenders.research.data.reco_dataset import get_reco_dataset
from generative_recommenders.research.indexing.utils import get_top_k_module
from generative_recommenders.research.modeling.sequential.autoregressive_losses import (
    BCELoss,
    InBatchNegativesSampler,
    LocalNegativesSampler,
)
from generative_recommenders.research.modeling.sequential.embedding_modules import (
    EmbeddingModule,
    LocalEmbeddingModule,
)
from generative_recommenders.research.modeling.sequential.encoder_utils import (
    get_sequential_encoder,
)
from generative_recommenders.research.modeling.sequential.features import (
    movielens_seq_features_from_row,
)
from generative_recommenders.research.modeling.sequential.input_features_preprocessors import (
    LearnablePositionalEmbeddingInputFeaturesPreprocessor,
)
from generative_recommenders.research.modeling.sequential.losses.sampled_softmax import (
    SampledSoftmaxLoss,
)
from generative_recommenders.research.modeling.sequential.output_postprocessors import (
    L2NormEmbeddingPostprocessor,
    LayerNormEmbeddingPostprocessor,
)
from generative_recommenders.research.modeling.similarity_utils import (
    get_similarity_function,
)
from generative_recommenders.research.trainer.data_loader import create_data_loader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter


def setup(rank: int, world_size: int, master_port: int) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup() -> None:
    dist.destroy_process_group()


@gin.configurable
def get_weighted_loss(
    main_loss: torch.Tensor,
    aux_losses: Dict[str, torch.Tensor],
    weights: Dict[str, float],
) -> torch.Tensor:
    weighted_loss = main_loss
    for key, weight in weights.items():
        cur_weighted_loss = aux_losses[key] * weight
        weighted_loss = weighted_loss + cur_weighted_loss
    return weighted_loss


@gin.configurable
def train_fn(
    rank: int,
    world_size: int,
    master_port: int,
    dataset_name: str = "ml-20m",
    max_sequence_length: int = 200,
    positional_sampling_ratio: float = 1.0,
    local_batch_size: int = 128,
    eval_batch_size: int = 128,
    eval_user_max_batch_size: Optional[int] = None,
    main_module: str = "SASRec",
    main_module_bf16: bool = False,
    dropout_rate: float = 0.2,
    user_embedding_norm: str = "l2_norm",
    sampling_strategy: str = "in-batch",
    loss_module: str = "SampledSoftmaxLoss",
    loss_weights: Optional[Dict[str, float]] = {},
    num_negatives: int = 1,
    loss_activation_checkpoint: bool = False,
    item_l2_norm: bool = False,
    temperature: float = 0.05,
    num_epochs: int = 101,
    learning_rate: float = 1e-3,
    num_warmup_steps: int = 0,
    weight_decay: float = 1e-3,
    top_k_method: str = "MIPSBruteForceTopK",
    eval_interval: int = 100,
    full_eval_every_n: int = 1,
    save_ckpt_every_n: int = 1000,
    partial_eval_num_iters: int = 32,
    embedding_module_type: str = "local",
    item_embedding_dim: int = 240,
    interaction_module_type: str = "",
    gr_output_length: int = 10,
    l2_norm_eps: float = 1e-6,
    enable_tf32: bool = False,
    random_seed: int = 42,
) -> None:
    # to enable more deterministic results.
    random.seed(random_seed)
    torch.backends.cuda.matmul.allow_tf32 = enable_tf32
    torch.backends.cudnn.allow_tf32 = enable_tf32
    logging.info(f"cuda.matmul.allow_tf32: {enable_tf32}")
    logging.info(f"cudnn.allow_tf32: {enable_tf32}")
    logging.info(f"Training model on rank {rank}.")
    setup(rank, world_size, master_port)

    dataset = get_reco_dataset(
        dataset_name=dataset_name,
        max_sequence_length=max_sequence_length,
        chronological=True,
        positional_sampling_ratio=positional_sampling_ratio,
    )

    train_data_sampler, train_data_loader = create_data_loader(
        dataset.train_dataset,
        batch_size=local_batch_size,
        world_size=world_size,
        rank=rank,
        shuffle=True,
        drop_last=world_size > 1,
    )
    eval_data_sampler, eval_data_loader = create_data_loader(
        dataset.eval_dataset,
        batch_size=eval_batch_size,
        world_size=world_size,
        rank=rank,
        shuffle=True,  # needed for partial eval
        drop_last=world_size > 1,
    )

    model_debug_str = main_module
    if embedding_module_type == "local":
        embedding_module: EmbeddingModule = LocalEmbeddingModule(
            num_items=dataset.max_item_id,
            item_embedding_dim=item_embedding_dim,
        )
    else:
        raise ValueError(f"Unknown embedding_module_type {embedding_module_type}")
    model_debug_str += f"-{embedding_module.debug_str()}"

    interaction_module, interaction_module_debug_str = get_similarity_function(
        module_type=interaction_module_type,
        query_embedding_dim=item_embedding_dim,
        item_embedding_dim=item_embedding_dim,
    )

    assert user_embedding_norm == "l2_norm" or user_embedding_norm == "layer_norm", (
        f"Not implemented for {user_embedding_norm}"
    )
    output_postproc_module = (
        L2NormEmbeddingPostprocessor(
            embedding_dim=item_embedding_dim,
            eps=1e-6,
        )
        if user_embedding_norm == "l2_norm"
        else LayerNormEmbeddingPostprocessor(
            embedding_dim=item_embedding_dim,
            eps=1e-6,
        )
    )
    input_preproc_module = LearnablePositionalEmbeddingInputFeaturesPreprocessor(
        max_sequence_len=dataset.max_sequence_length + gr_output_length + 1,
        embedding_dim=item_embedding_dim,
        dropout_rate=dropout_rate,
    )

    model = get_sequential_encoder(
        module_type=main_module,
        max_sequence_length=dataset.max_sequence_length,
        max_output_length=gr_output_length + 1,
        embedding_module=embedding_module,
        interaction_module=interaction_module,
        input_preproc_module=input_preproc_module,
        output_postproc_module=output_postproc_module,
        verbose=True,
    )
    model_debug_str = model.debug_str()

    # loss
    loss_debug_str = loss_module
    if loss_module == "BCELoss":
        loss_debug_str = loss_debug_str[:-4]
        assert temperature == 1.0
        ar_loss = BCELoss(temperature=temperature, model=model)
    elif loss_module == "SampledSoftmaxLoss":
        loss_debug_str = "ssl"
        if temperature != 1.0:
            loss_debug_str += f"-t{temperature}"
        ar_loss = SampledSoftmaxLoss(
            num_to_sample=num_negatives,
            softmax_temperature=temperature,
            model=model,
            activation_checkpoint=loss_activation_checkpoint,
        )
        loss_debug_str += (
            f"-n{num_negatives}{'-ac' if loss_activation_checkpoint else ''}"
        )
    else:
        raise ValueError(f"Unrecognized loss module {loss_module}.")

    # sampling
    if sampling_strategy == "in-batch":
        negatives_sampler = InBatchNegativesSampler(
            l2_norm=item_l2_norm,
            l2_norm_eps=l2_norm_eps,
            dedup_embeddings=True,
        )
        sampling_debug_str = (
            f"in-batch{f'-l2-eps{l2_norm_eps}' if item_l2_norm else ''}-dedup"
        )
    elif sampling_strategy == "local":
        negatives_sampler = LocalNegativesSampler(
            num_items=dataset.max_item_id,
            item_emb=model._embedding_module._item_emb,
            all_item_ids=dataset.all_item_ids,
            l2_norm=item_l2_norm,
            l2_norm_eps=l2_norm_eps,
        )
    else:
        raise ValueError(f"Unrecognized sampling strategy {sampling_strategy}.")
    sampling_debug_str = negatives_sampler.debug_str()

    # Creates model and moves it to GPU with id rank
    device = rank
    if main_module_bf16:
        model = model.to(torch.bfloat16)
    model = model.to(device)
    ar_loss = ar_loss.to(device)
    negatives_sampler = negatives_sampler.to(device)
    model = DDP(model, device_ids=[rank], broadcast_buffers=False)

    # TODO: wrap in create_optimizer.
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.98),
        weight_decay=weight_decay,
    )

    date_str = date.today().strftime("%Y-%m-%d")
    model_subfolder = f"{dataset_name}-l{max_sequence_length}"
    model_desc = (
        f"{model_subfolder}"
        + f"/{model_debug_str}_{interaction_module_debug_str}_{sampling_debug_str}_{loss_debug_str}"
        + f"{f'-ddp{world_size}' if world_size > 1 else ''}-b{local_batch_size}-lr{learning_rate}-wu{num_warmup_steps}-wd{weight_decay}{'' if enable_tf32 else '-notf32'}-{date_str}"
    )
    if full_eval_every_n > 1:
        model_desc += f"-fe{full_eval_every_n}"
    if positional_sampling_ratio is not None and positional_sampling_ratio < 1:
        model_desc += f"-d{positional_sampling_ratio}"
    # creates subfolders.
    os.makedirs(f"./exps/{model_subfolder}", exist_ok=True)
    os.makedirs(f"./ckpts/{model_subfolder}", exist_ok=True)
    log_dir = f"./exps/{model_desc}"
    if rank == 0:
        writer = SummaryWriter(log_dir=log_dir)
        logging.info(f"Rank {rank}: writing logs to {log_dir}")
    else:
        writer = None
        logging.info(f"Rank {rank}: disabling summary writer")

    last_training_time = time.time()
    torch.autograd.set_detect_anomaly(True)

    batch_id = 0
    epoch = 0
    for epoch in range(num_epochs):
        if train_data_sampler is not None:
            train_data_sampler.set_epoch(epoch)
        if eval_data_sampler is not None:
            eval_data_sampler.set_epoch(epoch)
        model.train()
        for row in iter(train_data_loader):
            seq_features, target_ids, target_ratings = movielens_seq_features_from_row(
                row,
                device=device,
                max_output_length=gr_output_length + 1,
            )

            if (batch_id % eval_interval) == 0:
                model.eval()

                eval_state = get_eval_state(
                    model=model.module,
                    all_item_ids=dataset.all_item_ids,
                    negatives_sampler=negatives_sampler,
                    top_k_module_fn=lambda item_embeddings, item_ids: get_top_k_module(
                        top_k_method=top_k_method,
                        model=model.module,
                        item_embeddings=item_embeddings,
                        item_ids=item_ids,
                    ),
                    device=device,
                    float_dtype=torch.bfloat16 if main_module_bf16 else None,
                )
                eval_dict = eval_metrics_v2_from_tensors(
                    eval_state,
                    model.module,
                    seq_features,
                    target_ids=target_ids,
                    target_ratings=target_ratings,
                    user_max_batch_size=eval_user_max_batch_size,
                    dtype=torch.bfloat16 if main_module_bf16 else None,
                )
                add_to_summary_writer(
                    writer, batch_id, eval_dict, prefix="eval", world_size=world_size
                )
                logging.info(
                    f"rank {rank}:  batch-stat (eval): iter {batch_id} (epoch {epoch}): "
                    + f"NDCG@10 {_avg(eval_dict['ndcg@10'], world_size):.4f}, "
                    f"HR@10 {_avg(eval_dict['hr@10'], world_size):.4f}, "
                    f"HR@50 {_avg(eval_dict['hr@50'], world_size):.4f}, "
                    + f"MRR {_avg(eval_dict['mrr'], world_size):.4f} "
                )
                model.train()

            # TODO: consider separating this out?
            B, N = seq_features.past_ids.shape
            seq_features.past_ids.scatter_(
                dim=1,
                index=seq_features.past_lengths.view(-1, 1),
                src=target_ids.view(-1, 1),
            )

            opt.zero_grad()
            input_embeddings = model.module.get_item_embeddings(seq_features.past_ids)
            seq_embeddings = model(
                past_lengths=seq_features.past_lengths,
                past_ids=seq_features.past_ids,
                past_embeddings=input_embeddings,
                past_payloads=seq_features.past_payloads,
            )  # [B, X]

            supervision_ids = seq_features.past_ids

            if sampling_strategy == "in-batch":
                # get_item_embeddings currently assume 1-d tensor.
                in_batch_ids = supervision_ids.view(-1)
                negatives_sampler.process_batch(
                    ids=in_batch_ids,
                    presences=(in_batch_ids != 0),
                    embeddings=model.module.get_item_embeddings(in_batch_ids),
                )
            else:
                # pyre-fixme[16]: `InBatchNegativesSampler` has no attribute
                #  `_item_emb`.
                negatives_sampler._item_emb = model.module._embedding_module._item_emb

            ar_mask = supervision_ids[:, 1:] != 0
            loss, aux_losses = ar_loss(
                lengths=seq_features.past_lengths,  # [B],
                output_embeddings=seq_embeddings[:, :-1, :],  # [B, N-1, D]
                supervision_ids=supervision_ids[:, 1:],  # [B, N-1]
                supervision_embeddings=input_embeddings[:, 1:, :],  # [B, N - 1, D]
                supervision_weights=ar_mask.float(),
                negatives_sampler=negatives_sampler,
                **seq_features.past_payloads,
            )  # [B, N]

            main_loss = loss.detach().clone()
            loss = get_weighted_loss(loss, aux_losses, weights=loss_weights or {})

            if rank == 0:
                assert writer is not None
                writer.add_scalar("losses/ar_loss", loss, batch_id)
                writer.add_scalar("losses/main_loss", main_loss, batch_id)

            loss.backward()

            # Optional linear warmup.
            if batch_id < num_warmup_steps:
                lr_scalar = min(1.0, float(batch_id + 1) / num_warmup_steps)
                for pg in opt.param_groups:
                    pg["lr"] = lr_scalar * learning_rate
                lr = lr_scalar * learning_rate
            else:
                lr = learning_rate

            if (batch_id % eval_interval) == 0:
                logging.info(
                    f" rank: {rank}, batch-stat (train): step {batch_id} "
                    f"(epoch {epoch} in {time.time() - last_training_time:.2f}s): {loss:.6f}"
                )
                last_training_time = time.time()
                if rank == 0:
                    assert writer is not None
                    writer.add_scalar("loss/train", loss, batch_id)
                    writer.add_scalar("lr", lr, batch_id)

            opt.step()

            batch_id += 1

        def is_full_eval(epoch: int) -> bool:
            return (epoch % full_eval_every_n) == 0

        # eval per epoch
        eval_dict_all = None
        eval_start_time = time.time()
        model.eval()
        eval_state = get_eval_state(
            model=model.module,
            all_item_ids=dataset.all_item_ids,
            negatives_sampler=negatives_sampler,
            top_k_module_fn=lambda item_embeddings, item_ids: get_top_k_module(
                top_k_method=top_k_method,
                model=model.module,
                item_embeddings=item_embeddings,
                item_ids=item_ids,
            ),
            device=device,
            float_dtype=torch.bfloat16 if main_module_bf16 else None,
        )
        for eval_iter, row in enumerate(iter(eval_data_loader)):
            seq_features, target_ids, target_ratings = movielens_seq_features_from_row(
                row, device=device, max_output_length=gr_output_length + 1
            )
            eval_dict = eval_metrics_v2_from_tensors(
                eval_state,
                model.module,
                seq_features,
                target_ids=target_ids,
                target_ratings=target_ratings,
                user_max_batch_size=eval_user_max_batch_size,
                dtype=torch.bfloat16 if main_module_bf16 else None,
            )

            if eval_dict_all is None:
                eval_dict_all = {}
                for k in eval_dict.keys():
                    eval_dict_all[k] = []

            for k, v in eval_dict.items():
                eval_dict_all[k] = eval_dict_all[k] + [v]
            del eval_dict

            if (eval_iter + 1 >= partial_eval_num_iters) and (not is_full_eval(epoch)):
                logging.info(
                    f"Truncating epoch {epoch} eval to {eval_iter + 1} iters to save cost.."
                )
                break

        assert eval_dict_all is not None
        for k, v in eval_dict_all.items():
            eval_dict_all[k] = torch.cat(v, dim=-1)

        ndcg_10 = _avg(eval_dict_all["ndcg@10"], world_size=world_size)
        ndcg_50 = _avg(eval_dict_all["ndcg@50"], world_size=world_size)
        hr_10 = _avg(eval_dict_all["hr@10"], world_size=world_size)
        hr_50 = _avg(eval_dict_all["hr@50"], world_size=world_size)
        mrr = _avg(eval_dict_all["mrr"], world_size=world_size)

        add_to_summary_writer(
            writer,
            batch_id=epoch,
            metrics=eval_dict_all,
            prefix="eval_epoch",
            world_size=world_size,
        )
        if full_eval_every_n > 1 and is_full_eval(epoch):
            add_to_summary_writer(
                writer,
                batch_id=epoch,
                metrics=eval_dict_all,
                prefix="eval_epoch_full",
                world_size=world_size,
            )
        if rank == 0 and epoch > 0 and (epoch % save_ckpt_every_n) == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                },
                f"./ckpts/{model_desc}_ep{epoch}",
            )

        logging.info(
            f"rank {rank}: eval @ epoch {epoch} in {time.time() - eval_start_time:.2f}s: "
            f"NDCG@10 {ndcg_10:.4f}, NDCG@50 {ndcg_50:.4f}, HR@10 {hr_10:.4f}, HR@50 {hr_50:.4f}, MRR {mrr:.4f}"
        )
        last_training_time = time.time()

    if rank == 0:
        if writer is not None:
            writer.flush()
            writer.close()

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
            },
            f"./ckpts/{model_desc}_ep{epoch}",
        )

    cleanup()
