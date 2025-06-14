# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# This file is modified from https://github.com/haotian-liu/LLaVA/


import os
import random
from collections import OrderedDict
from typing import List, Optional, Dict, Union, Any

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.data import ConcatDataset, Dataset, DistributedSampler, RandomSampler, Sampler
from transformers import PreTrainedModel, Trainer
from transformers.modeling_utils import unwrap_model
from transformers.trainer import ALL_LAYERNORM_LAYERS  # ShardedDDPOption,
from transformers.trainer import get_parameter_names, has_length, is_sagemaker_mp_enabled, logger

from llava.train.sequence_parallel import get_pg_manager
# from llava.trl.trainer import DPOTrainer


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [
        lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)
    ]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) > 0:
        megabatches.append(sorted(additional_batch))

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class VILADistributedSampler(DistributedSampler):
    """This class is implemented by Jason Lu."""

    def __init__(
        self,
        dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        batch_size=None,
        # NOTE: this is the total size but not per-worker
        sample_len_list=None,
        force_accumulation=True,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval" " [0, {}]".format(rank, num_replicas - 1)
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = True  # always True

        # NOTE: org_ is without drop last
        self.org_sample_len_list = self.per_replica_samples = sample_len_list
        assert sum(sample_len_list) == len(self.dataset)

        self.batch_size = batch_size
        self.global_batch_size = batch_size * num_replicas

        if self.drop_last:  # type: ignore[arg-type]
            self.per_replica_samples = [
                sample_len // (self.num_replicas * batch_size) * batch_size for sample_len in self.per_replica_samples
            ]
            self.num_samples = sum(self.per_replica_samples)
        else:
            raise NotImplementedError

        self.total_size = self.num_samples * self.num_replicas
        self.total_samples = [samples * self.num_replicas for samples in self.per_replica_samples]

        self.shuffle = shuffle
        self.seed = seed

        # whether to force accumulate
        self.force_accumulation = force_accumulation

    def __iter__(self):

        indices = list(range(len(self.dataset)))

        # 1. split the full indices first (note: without drop last at this moment)
        indices_list = []
        for i in range(len(self.org_sample_len_list)):
            indices_list.append(
                indices[sum(self.org_sample_len_list[:i]) : sum(self.org_sample_len_list[:i]) + self.total_samples[i]]
            )

        assert sum([len(indices) for indices in indices_list]) == self.total_size, (
            sum([len(indices) for indices in indices_list]),
            self.total_size,
        )

        # let's first do subsample
        for idx, indices in enumerate(indices_list):
            indices_list[idx] = indices[
                self.rank * self.per_replica_samples[idx] : (self.rank + 1) * self.per_replica_samples[idx]
            ]

        random.seed(self.seed + self.epoch)
        for indice in range(len(indices_list)):
            random.shuffle(indices_list[indice])

        indices_list = sorted(indices_list, key=lambda x: -len(x))
        all_indices = [-1] * self.num_samples
        indices_available = list(range(self.num_samples))
        for indice in indices_list:
            original_indices = range(len(indice))
            transformed_indices = [idx * len(indices_available) // len(indice) for idx in original_indices]
            mapped_indices = [indices_available[idx] for idx in transformed_indices]
            # update indices_available
            for idx in reversed(transformed_indices):
                del indices_available[idx]
            for i, idx in enumerate(mapped_indices):
                all_indices[idx] = indice[i]
        assert -1 not in all_indices

        return iter(all_indices)

    # # (Qinghao): Implementation for validating accuracy of SP
    # def __iter__(self):
    #     iterator = super().__iter__()
    #     indices = list(iterator)
    #     indices = indices[self.start_index :]
    #     return iter(indices)

    # def __len__(self) -> int:
    #     return self.num_samples - self.start_index

    # def set_start_index(self, start_index: int) -> None:
    #     self.start_index = start_index


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        else:
            indices = get_length_grouped_indices(
                self.lengths, self.batch_size, self.world_size, generator=self.generator
            )
        return iter(indices)


# class VILADPOTrainer(DPOTrainer):
#     def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
#         if self.train_dataset is None or not has_length(self.train_dataset):
#             return None

#         # Always using Jason's sampler.
#         sample_len_list = self.args.sample_lens
#         seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
#         num_replicas = self.args.world_size
#         rank = self.args.process_index

#         # Consider sequence parallelism
#         sp_degree = self.args.seq_parallel_size
#         if sp_degree > 1:  # Sequence Parallelism is enabled
#             num_replicas = num_replicas // sp_degree
#             PROCESS_GROUP_MANAGER = get_pg_manager()
#             rank = PROCESS_GROUP_MANAGER.dp_rank
#             # rank = dist.get_rank() // sp_degree

#         return VILADistributedSampler(
#             self.train_dataset,
#             num_replicas=num_replicas,
#             rank=rank,
#             seed=seed,
#             batch_size=self.args.train_batch_size,
#             sample_len_list=sample_len_list,
#         )

#         # if self.args.group_by_modality_length:
#         #     if not isinstance(self.train_dataset, ConcatDataset):
#         #         lengths = self.train_dataset.modality_lengths
#         #     else:
#         #         lengths = []
#         #         for d in self.train_dataset.datasets:
#         #             lengths += d.modality_lengths
#         #     return LengthGroupedSampler(
#         #         self.args.train_batch_size,
#         #         world_size=self.args.world_size * self.args.gradient_accumulation_steps,
#         #         lengths=lengths,
#         #         group_by_modality=True,
#         #     )
#         # else:
#         #     return super()._get_train_sampler()

#     def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[torch.utils.data.Sampler]:
#         if self.eval_dataset is None or not has_length(self.eval_dataset):
#             return None

#         # Always using Jason's sampler.
#         sample_len_list = self.args.eval_sample_lens
#         seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
#         return VILADistributedSampler(
#             eval_dataset,
#             num_replicas=self.args.world_size,
#             rank=self.args.process_index,
#             seed=seed,
#             batch_size=self.args.eval_batch_size,
#             sample_len_list=sample_len_list,
#         )

#     def create_optimizer(self):
#         """
#         Setup the optimizer.

#         We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
#         Trainer's init through `optimizers`, or subclass and override this method in a subclass.
#         """
#         if is_sagemaker_mp_enabled():
#             return super().create_optimizer()
#         # if self.sharded_ddp == ShardedDDPOption.SIMPLE:
#         #     return super().create_optimizer()

#         opt_model = self.model

#         if self.optimizer is None:
#             decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
#             decay_parameters = [name for name in decay_parameters if "bias" not in name]
#             if self.args.mm_projector_lr is not None:
#                 projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
#                 optimizer_grouped_parameters = [
#                     {
#                         "params": [
#                             p
#                             for n, p in opt_model.named_parameters()
#                             if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
#                         ],
#                         "weight_decay": self.args.weight_decay,
#                     },
#                     {
#                         "params": [
#                             p
#                             for n, p in opt_model.named_parameters()
#                             if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
#                         ],
#                         "weight_decay": 0.0,
#                     },
#                     {
#                         "params": [
#                             p
#                             for n, p in opt_model.named_parameters()
#                             if (n in decay_parameters and n in projector_parameters and p.requires_grad)
#                         ],
#                         "weight_decay": self.args.weight_decay,
#                         "lr": self.args.mm_projector_lr,
#                     },
#                     {
#                         "params": [
#                             p
#                             for n, p in opt_model.named_parameters()
#                             if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
#                         ],
#                         "weight_decay": 0.0,
#                         "lr": self.args.mm_projector_lr,
#                     },
#                 ]
#             else:
#                 optimizer_grouped_parameters = [
#                     {
#                         "params": [
#                             p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
#                         ],
#                         "weight_decay": self.args.weight_decay,
#                     },
#                     {
#                         "params": [
#                             p
#                             for n, p in opt_model.named_parameters()
#                             if (n not in decay_parameters and p.requires_grad)
#                         ],
#                         "weight_decay": 0.0,
#                     },
#                 ]

#             optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

#             if 0:  # self.sharded_ddp == ShardedDDPOption.SIMPLE:
#                 self.optimizer = OSS(
#                     params=optimizer_grouped_parameters,
#                     optim=optimizer_cls,
#                     **optimizer_kwargs,
#                 )
#             else:
#                 self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
#                 if optimizer_cls.__name__ == "Adam8bit":
#                     import bitsandbytes

#                     manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

#                     skipped = 0
#                     for module in opt_model.modules():
#                         if isinstance(module, nn.Embedding):
#                             skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
#                             logger.info(f"skipped {module}: {skipped/2**20}M params")
#                             manager.register_module_override(module, "weight", {"optim_bits": 32})
#                             logger.debug(f"bitsandbytes: will optimize {module} in fp32")
#                     logger.info(f"skipped: {skipped/2**20}M params")

#         return self.optimizer

#     def save_model(self, output_dir: Optional[str], _internal_call: bool):
#         ## save tuned model separately
#         if self.is_deepspeed_enabled:
#             state_dict = self.accelerator.get_state_dict(self.deepspeed)
#         else:
#             # TODO(ligeng): fix save_model for multi-node training on large models (e.g., Llama-70b)
#             state_dict = self.model.state_dict()

#         if self.args.should_save:
#             return self.model.save_pretrained(output_dir, state_dict=state_dict)


class LLaVATrainer(Trainer):
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # Always using Jason's sampler.
        sample_len_list = self.args.sample_lens
        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
        num_replicas = self.args.world_size
        rank = self.args.process_index

        # Consider sequence parallelism
        sp_degree = self.args.seq_parallel_size
        if sp_degree > 1:  # Sequence Parallelism is enabled
            num_replicas = num_replicas // sp_degree
            PROCESS_GROUP_MANAGER = get_pg_manager()
            rank = PROCESS_GROUP_MANAGER.dp_rank
            # rank = dist.get_rank() // sp_degree

        return VILADistributedSampler(
            self.train_dataset,
            num_replicas=num_replicas,
            rank=rank,
            seed=seed,
            batch_size=self.args.train_batch_size,
            sample_len_list=sample_len_list,
        )

        # if self.args.group_by_modality_length:
        #     if not isinstance(self.train_dataset, ConcatDataset):
        #         lengths = self.train_dataset.modality_lengths
        #     else:
        #         lengths = []
        #         for d in self.train_dataset.datasets:
        #             lengths += d.modality_lengths
        #     return LengthGroupedSampler(
        #         self.args.train_batch_size,
        #         world_size=self.args.world_size * self.args.gradient_accumulation_steps,
        #         lengths=lengths,
        #         group_by_modality=True,
        #     )
        # else:
        #     return super()._get_train_sampler()

    def _get_eval_sampler(self, eval_dataset: Dataset) -> Optional[torch.utils.data.Sampler]:
        if self.eval_dataset is None or not has_length(self.eval_dataset):
            return None

        # Always using Jason's sampler.
        sample_len_list = self.args.eval_sample_lens
        seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
        return VILADistributedSampler(
            eval_dataset,
            num_replicas=self.args.world_size,
            rank=self.args.process_index,
            seed=seed,
            batch_size=self.args.eval_batch_size,
            sample_len_list=sample_len_list,
        )

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
        # if self.sharded_ddp == ShardedDDPOption.SIMPLE:
        #     return super().create_optimizer()

        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [name for name in decay_parameters if "bias" not in name]
            if self.args.mm_projector_lr is not None:
                projector_parameters = [name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n not in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p
                            for n, p in opt_model.named_parameters()
                            if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            if 0:  # self.sharded_ddp == ShardedDDPOption.SIMPLE:
                self.optimizer = OSS(
                    params=optimizer_grouped_parameters,
                    optim=optimizer_cls,
                    **optimizer_kwargs,
                )
            else:
                self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                if optimizer_cls.__name__ == "Adam8bit":
                    import bitsandbytes

                    manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                    skipped = 0
                    for module in opt_model.modules():
                        if isinstance(module, nn.Embedding):
                            skipped += sum({p.data_ptr(): p.numel() for p in module.parameters()}.values())
                            logger.info(f"skipped {module}: {skipped/2**20}M params")
                            manager.register_module_override(module, "weight", {"optim_bits": 32})
                            logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                    logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def save_model(self, output_dir: Optional[str], _internal_call: bool):
        ## save tuned model separately
        if self.is_deepspeed_enabled:
            state_dict = self.accelerator.get_state_dict(self.deepspeed)
        else:
            # TODO(ligeng): fix save_model for multi-node training on large models (e.g., Llama-70b)
            state_dict = self.model.state_dict()

        if self.args.lora_enable:
            non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(self.model.named_parameters())
            os.makedirs(output_dir, exist_ok=True)
            torch.save(
                non_lora_state_dict,
                os.path.join(output_dir, "non_lora_trainables.bin"),
            )

        if self.args.should_save:
            return self.model.save_pretrained(output_dir, state_dict=state_dict)

    # In class LLaVATrainer(Trainer)

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Override the standard training step to explicitly log custom losses.
        """
        # Use this print statement to be 100% sure your custom method is being called.
        # print("--- Executing Custom LLaVATrainer training_step ---")
    
        model.train()
        inputs = self._prepare_inputs(inputs)
    
        with self.compute_loss_context_manager():
            # The return_outputs=True flag is essential.
            # It gives us the full output object from the model, not just the loss tensor.
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
    
        if self.args.n_gpu > 1:
            loss = loss.mean()
    
        # --- START: Explicit Logging Logic ---
        # We create a dictionary of logs and pass it to self.log()
        # This is the standard Hugging Face Trainer way to log metrics.
        logs: Dict[str, float] = {}
    
        # The loss returned by `training_step` is used for logging the training loss
        logs["total_loss"] = loss.detach().item()
    
        # Check for our custom losses on the outputs object and add them to the logs
        if hasattr(outputs, "region_loss"):
            logs["region_loss"] = outputs.region_loss.item()
        
        # You can add more logs for other heads here in the future
        # if hasattr(outputs, "distance_loss"):
        #     logs["distance_loss"] = outputs.distance_loss.item()
    
        # Log only on the main process
        if self.state.is_local_process_zero:
            self.log(logs)
        # --- END: Explicit Logging Logic ---
    
        self.accelerator.backward(loss)
    
        return loss.detach() / self.args.gradient_accumulation_steps

        
    # override the compute_loss for head losses
    # def compute_loss(self, model, inputs, return_outputs=False):
    #     # get th standard llm loss
    #     llm_loss, outputs =  super().compute_loss(model, inputs, return_outputs)
        
    #     # get the region_loss
    #     region_logits = getattr(self, 'region_logits', None)
    #     if region_logits is None:
    #         return llm_loss if not return_outputs else (llm_loss, outputs)
        
    #     region_labels = inputs.get('region_labels')
            
    #     region_loss = self.region_classifier_loss(region_logits, region_labels)
        
    #     # Combine losses
    #     loss_weights = {
    #         'llm': 1,
    #         'region_cls': 0.2
    #     }
    #     total_loss = (loss_weights['llm'] * llm_loss) + (loss_weights["region_cls"]*region_loss)
        
    #     logger.log({
    #         'llm_loss': llm_loss.item(),
    #         'region_loss': region_loss.item(),
    #         'total_loss': total_loss.item()
    #     })
        
    #     return total_loss if not return_outputs else (total_loss, outputs)


        
        
            
