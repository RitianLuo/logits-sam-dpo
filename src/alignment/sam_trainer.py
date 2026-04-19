from __future__ import annotations

from contextlib import nullcontext
from typing import Any, Union

import deepspeed
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast
from transformers import PreTrainedModel
from trl.trainer.dpo_trainer import DPOTrainer
from trl.trainer.utils import flush_left, flush_right, selective_log_softmax


MODEL_INPUT_KEYS = (
    "pixel_values",
    "pixel_attention_mask",
    "image_sizes",
    "image_grid_thw",
    "image_position_ids",
    "mm_token_type_ids",
)


def resolve_lm_head(model, accelerator=None):
    if accelerator is not None:
        base = accelerator.unwrap_model(model)
    else:
        try:
            from accelerate.utils import unwrap_model

            base = unwrap_model(model)
        except Exception:
            base = model.module if hasattr(model, "module") else model

    try:
        from peft import PeftModel

        if isinstance(base, PeftModel):
            base = base.get_base_model()
    except Exception:
        pass

    lm_head = (
        base.get_output_embeddings()
        if hasattr(base, "get_output_embeddings")
        else getattr(base, "lm_head", None)
    )
    if lm_head is None or not hasattr(lm_head, "weight"):
        raise AttributeError("Output head not found. The model must implement get_output_embeddings() or expose lm_head.")
    return base, lm_head


def resolve_model_body(model, accelerator=None):
    base, lm_head = resolve_lm_head(model, accelerator=accelerator)

    if hasattr(base, "get_decoder"):
        try:
            decoder = base.get_decoder()
            if isinstance(decoder, nn.Module):
                return base, decoder, lm_head
        except Exception:
            pass

    for attr in ("model", "language_model", "transformer"):
        candidate = getattr(base, attr, None)
        if isinstance(candidate, nn.Module):
            return base, candidate, lm_head

    return base, base, lm_head


def _is_zero3(accelerator=None):
    try:
        plug = getattr(getattr(accelerator, "state", None), "deepspeed_plugin", None)
        return getattr(plug, "zero_stage", 0) == 3
    except Exception:
        return False


class LogitsSAMTrainer(DPOTrainer):
    """
    DPO trainer with weight-only logits-SAM on the output head.

    The forward path stops at the decoder hidden states and reuses them to
    build both clean and perturbed logits.

    For ZeRO-3:
    - adaptive mode keeps the manual GatheredParameters path
    - non-adaptive mode uses the real lm_head.forward() and adds a chunked
      delta_logits term without gathering the head parameters
    - bias is never perturbed
    """

    def __init__(
        self,
        *args,
        sam_rho: float = 0.05,
        sam_adaptive: bool = False,
        sam_use_chunked_delta_logits: bool = False,
        sam_vocab_chunk_size: int = 4096,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.sam_rho = sam_rho
        self.sam_adaptive = sam_adaptive
        self.sam_use_chunked_delta_logits = sam_use_chunked_delta_logits
        if sam_vocab_chunk_size <= 0:
            raise ValueError("sam_vocab_chunk_size must be > 0.")
        self.sam_vocab_chunk_size = sam_vocab_chunk_size

    def _prepare_decoder_batch(
        self, batch: dict[str, Union[list, torch.LongTensor]]
    ) -> dict[str, torch.Tensor | int | None]:
        if self.is_encoder_decoder:
            raise NotImplementedError("LogitsSAMTrainer currently supports decoder-only models only.")

        num_examples = batch["prompt_input_ids"].shape[0]
        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)

        model_inputs = {"use_cache": False}
        if self.aux_loss_enabled:
            model_inputs["output_router_logits"] = True

        for key in MODEL_INPUT_KEYS:
            if key in concatenated_batch:
                model_inputs[key] = concatenated_batch[key]

        if "prompt_token_type_ids" in concatenated_batch and "completion_token_type_ids" in concatenated_batch:
            model_inputs["token_type_ids"] = torch.cat(
                (concatenated_batch["prompt_token_type_ids"], concatenated_batch["completion_token_type_ids"]), dim=1
            )
        elif "token_type_ids" in concatenated_batch:
            model_inputs["token_type_ids"] = concatenated_batch["token_type_ids"]

        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]

        input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
        attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
        loss_mask = torch.cat((torch.zeros_like(prompt_attention_mask), completion_attention_mask), dim=1)

        if self.max_length is not None and self.max_length < attention_mask.size(1):
            if self.truncation_mode == "keep_start":
                attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
                attention_mask = attention_mask[:, : self.max_length]
                input_ids = input_ids[:, : self.max_length]
                loss_mask = loss_mask[:, : self.max_length]
            elif self.truncation_mode == "keep_end":
                attention_mask, input_ids, loss_mask = flush_right(attention_mask, input_ids, loss_mask)
                input_ids = input_ids[:, -self.max_length :]
                attention_mask = attention_mask[:, -self.max_length :]
                loss_mask = loss_mask[:, -self.max_length :]
                attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
            else:
                raise ValueError(
                    f"Unknown truncation mode: '{self.truncation_mode}'. Should be one of ['keep_end', 'keep_start']."
                )
        else:
            attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)

        logits_to_keep = None
        if self.use_logits_to_keep:
            first_compute_index = loss_mask.nonzero(as_tuple=True)[1].min()
            logits_to_keep = (loss_mask.shape[1] - first_compute_index).item() + 1

        flat_attention_mask = None
        position_ids = None
        if self.padding_free:
            flat_attention_mask = attention_mask
            input_ids = input_ids[flat_attention_mask.bool()].unsqueeze(0)
            loss_mask = loss_mask[flat_attention_mask.bool()].unsqueeze(0)
            position_ids = flat_attention_mask.cumsum(1)[flat_attention_mask.bool()].unsqueeze(0) - 1
            model_inputs["position_ids"] = position_ids

            token_type_ids = model_inputs.get("token_type_ids")
            if isinstance(token_type_ids, torch.Tensor) and token_type_ids.shape == flat_attention_mask.shape:
                model_inputs["token_type_ids"] = token_type_ids[flat_attention_mask.bool()].unsqueeze(0)
        else:
            model_inputs["attention_mask"] = attention_mask

        labels = torch.roll(input_ids, shifts=-1, dims=1)
        loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

        if logits_to_keep is not None:
            labels = labels[:, -logits_to_keep:]
            loss_mask = loss_mask[:, -logits_to_keep:]

        return {
            "num_examples": num_examples,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "flat_attention_mask": flat_attention_mask,
            "loss_mask": loss_mask,
            "labels": labels,
            "position_ids": position_ids,
            "logits_to_keep": logits_to_keep,
            "model_inputs": model_inputs,
        }

    def _forward_to_hidden_states(
        self,
        model: nn.Module,
        prepared: dict[str, torch.Tensor | int | None],
    ) -> tuple[torch.Tensor, nn.Module, torch.Tensor | None]:
        _, model_body, lm_head = resolve_model_body(model, accelerator=self.accelerator)
        input_ids = prepared["input_ids"]
        model_inputs = dict(prepared["model_inputs"])

        if self.aux_loss_enabled:
            model_inputs["output_hidden_states"] = True
            outputs = model(input_ids, **model_inputs)
            hidden_states = outputs.hidden_states[-1]
            aux_loss = getattr(outputs, "aux_loss", None)
        else:
            outputs = model_body(input_ids, **model_inputs)
            hidden_states = getattr(outputs, "last_hidden_state", None)
            if hidden_states is None and isinstance(outputs, (tuple, list)) and outputs:
                hidden_states = outputs[0]
            aux_loss = getattr(outputs, "aux_loss", None)

        if hidden_states is None:
            raise RuntimeError("Failed to retrieve hidden states before lm_head.")

        logits_to_keep = prepared["logits_to_keep"]
        if logits_to_keep is not None:
            hidden_states = hidden_states[:, -logits_to_keep:, :]

        labels = prepared["labels"]
        if hidden_states.shape[:2] != labels.shape[:2]:
            hidden_states = hidden_states[:, -labels.shape[1] :, :]

        return hidden_states.contiguous(), lm_head, aux_loss

    def _compute_logits_from_hidden(
        self,
        hidden_states: torch.Tensor,
        lm_head: nn.Module,
        weight_delta: torch.Tensor | None = None,
    ) -> torch.Tensor:
        head_weight = lm_head.weight
        head_bias = getattr(lm_head, "bias", None)
        if weight_delta is not None:
            head_weight = head_weight + weight_delta
        return F.linear(hidden_states.to(head_weight.dtype, copy=False), head_weight, head_bias)

    def _compute_logits_from_weight_bias(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return F.linear(hidden_states.to(weight.dtype, copy=False), weight, bias)

    def _get_batch_output_from_logits(
        self,
        logits: torch.Tensor,
        prepared: dict[str, torch.Tensor | int | None],
        *,
        aux_loss: torch.Tensor | None = None,
        is_ref_model: bool = False,
    ) -> dict[str, torch.Tensor]:
        labels = prepared["labels"].clone()
        loss_mask = prepared["loss_mask"]
        num_examples = prepared["num_examples"]

        if logits.shape[:2] != labels.shape[:2]:
            logits = logits[:, -labels.shape[1] :]

        labels[~loss_mask] = 0
        flat_per_token_logps = selective_log_softmax(logits, labels)
        flat_per_token_logps[~loss_mask] = 0
        flat_per_token_logps = torch.roll(flat_per_token_logps, shifts=1, dims=1)

        if self.padding_free:
            attention_mask = prepared["flat_attention_mask"]
            batch_size, seq_len = attention_mask.shape
            per_token_logps = torch.zeros(batch_size, seq_len, device=logits.device, dtype=flat_per_token_logps.dtype)
            sequence_loss_mask = torch.zeros(batch_size, seq_len, device=logits.device, dtype=torch.bool)
            per_token_logps[attention_mask.bool()] = flat_per_token_logps
            sequence_loss_mask[attention_mask.bool()] = loss_mask
        else:
            per_token_logps = flat_per_token_logps
            sequence_loss_mask = loss_mask

        all_logps = per_token_logps[:, 1:].sum(-1)
        output = {}

        if self.use_weighting:
            with torch.no_grad():
                logprobs = F.log_softmax(logits, dim=-1)
                weights_adjustment_factor = torch.logsumexp(2 * logprobs, dim=-1)
                flat_adjusted = flat_per_token_logps - weights_adjustment_factor
                if self.padding_free:
                    attention_mask = prepared["flat_attention_mask"]
                    batch_size, seq_len = attention_mask.shape
                    adjusted = torch.zeros(batch_size, seq_len, device=logits.device, dtype=flat_adjusted.dtype)
                    adjusted[attention_mask.bool()] = flat_adjusted
                    all_weights = (adjusted * sequence_loss_mask).sum(-1) / sequence_loss_mask.sum(-1).clamp_min(1)
                else:
                    all_weights = (flat_adjusted * loss_mask).sum(-1) / loss_mask.sum(-1).clamp_min(1)
                chosen_weights = all_weights[:num_examples]
                rejected_weights = all_weights[num_examples:]
                output["policy_weights"] = torch.clamp(torch.exp(chosen_weights + rejected_weights), max=1)

        if self.args.rpo_alpha is not None or "sft" in self.loss_type:
            chosen_logits = logits[:num_examples, :-1]
            chosen_labels = labels[:num_examples, :-1]
            output["nll_loss"] = F.cross_entropy(
                torch.flatten(chosen_logits, end_dim=1), torch.flatten(chosen_labels), ignore_index=0
            )

        if "ipo" in self.loss_type:
            all_logps = all_logps / sequence_loss_mask.sum(-1).clamp_min(1)

        if self.args.ld_alpha is not None and not is_ref_model:
            completion_lengths = sequence_loss_mask.sum(dim=1)
            chosen_lengths = completion_lengths[:num_examples]
            rejected_lengths = completion_lengths[num_examples:]
            public_lengths = torch.min(chosen_lengths, rejected_lengths)
            public_lengths = torch.cat([public_lengths, public_lengths], dim=0)

            seq_len = per_token_logps.size(1)
            position_ids = torch.arange(seq_len, device=per_token_logps.device).expand_as(per_token_logps)

            ld_mask = position_ids < public_lengths.unsqueeze(1)
            mask = position_ids < completion_lengths.unsqueeze(1)

            front_mask = (ld_mask & mask).float()
            rear_mask = (~ld_mask & mask).float()
            front_logps = (per_token_logps * front_mask).sum(dim=1)
            rear_logps = (per_token_logps * rear_mask).sum(dim=1)
            all_logps = front_logps + self.args.ld_alpha * rear_logps

        output["chosen_logps"] = all_logps[:num_examples]
        output["rejected_logps"] = all_logps[num_examples:]

        if self.padding_free:
            position_ids = prepared["position_ids"]
            split_idx = (position_ids == 0).nonzero(as_tuple=True)[1][num_examples]
            mean_chosen_logits = logits[0, :split_idx][loss_mask[0, :split_idx]].mean()
            mean_rejected_logits = logits[0, split_idx:][loss_mask[0, split_idx:]].mean()
        else:
            mean_chosen_logits = logits[:num_examples][loss_mask[:num_examples]].mean()
            mean_rejected_logits = logits[num_examples:][loss_mask[num_examples:]].mean()

        output["mean_chosen_logits"] = mean_chosen_logits
        output["mean_rejected_logits"] = mean_rejected_logits

        if aux_loss is not None:
            output["aux_loss"] = aux_loss

        return output

    def concatenated_forward_with_logits(
        self,
        precomputed_logits: torch.Tensor,
        model: nn.Module,
        batch: dict[str, Union[list, torch.LongTensor]],
        is_ref_model: bool = False,
        aux_loss: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del model
        prepared = self._prepare_decoder_batch(batch)
        return self._get_batch_output_from_logits(
            precomputed_logits,
            prepared,
            aux_loss=aux_loss,
            is_ref_model=is_ref_model,
        )

    def _compute_weighted_losses(
        self,
        policy_output: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor | Any],
        loss_types: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        losses = 0.0
        chosen_rewards = 0.0
        rejected_rewards = 0.0

        for idx, loss_type in enumerate(loss_types):
            batch_losses, batch_chosen_rewards, batch_rejected_rewards = self.dpo_loss(
                policy_output["chosen_logps"],
                policy_output["rejected_logps"],
                batch["ref_chosen_logps"],
                batch["ref_rejected_logps"],
                loss_type,
                policy_output,
            )
            weight = self.loss_weights[idx] if self.loss_weights else 1.0
            losses = losses + batch_losses * weight
            chosen_rewards = chosen_rewards + batch_chosen_rewards * weight
            rejected_rewards = rejected_rewards + batch_rejected_rewards * weight

        if self.args.rpo_alpha is not None and "nll_loss" in policy_output:
            losses = losses + self.args.rpo_alpha * policy_output["nll_loss"]
        if self.use_weighting and "policy_weights" in policy_output:
            losses = losses * policy_output["policy_weights"]
        if self.aux_loss_enabled and "aux_loss" in policy_output:
            losses = losses + self.aux_loss_coef * policy_output["aux_loss"]

        return losses, chosen_rewards, rejected_rewards

    def _build_perturbed_logits_zero3(
        self,
        model: nn.Module,
        hidden_states: torch.Tensor,
        lm_head: nn.Module,
        prepared: dict[str, torch.Tensor | int | None],
        batch: dict[str, torch.Tensor | Any],
        loss_types: list[str],
        aux_loss: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        head_params = [lm_head.weight]
        if getattr(lm_head, "bias", None) is not None:
            head_params.append(lm_head.bias)

        with deepspeed.zero.GatheredParameters(head_params, fwd_module=model, modifier_rank=0, enabled=True):
            head_weight = lm_head.weight
            head_bias = getattr(lm_head, "bias", None)
            hidden_states_pre = hidden_states.detach()

            clean_logits_pre = self._compute_logits_from_weight_bias(
                hidden_states_pre,
                head_weight.detach(),
                head_bias.detach() if head_bias is not None else None,
            )
            clean_logits_pre = clean_logits_pre.detach().requires_grad_(True)

            policy_output_pre = self._get_batch_output_from_logits(
                clean_logits_pre,
                prepared,
                aux_loss=aux_loss,
            )
            losses_pre, _, _ = self._compute_weighted_losses(policy_output_pre, batch, loss_types)
            loss_pre = losses_pre.mean()

            (grad_logits,) = torch.autograd.grad(loss_pre, clean_logits_pre, retain_graph=False, create_graph=False)
            grad_logits_det = grad_logits.detach().reshape(-1, grad_logits.shape[-1])
            hidden_det = hidden_states_pre.reshape(-1, hidden_states_pre.shape[-1]).to(grad_logits_det.dtype, copy=False)
            grad = grad_logits_det.t().matmul(hidden_det)
            if self.sam_adaptive:
                grad = grad * head_weight.detach().abs().to(grad.dtype, copy=False)
            with torch.amp.autocast(self.accelerator.device.type, enabled=False):
                grad_norm = grad.float().norm(p=2).clamp_min(1e-12)
            perturbation = (grad * (self.sam_rho / grad_norm).to(grad.dtype)).to(head_weight.dtype).detach()
            perturbed_logits = self._compute_logits_from_weight_bias(hidden_states, head_weight + perturbation, head_bias)

        return perturbed_logits, loss_pre.detach()

    def _compute_nonadaptive_delta_logits_chunked(
        self,
        hidden_states: torch.Tensor,
        grad_logits: torch.Tensor,
        out_dtype: torch.dtype,
        chunk_size: int | None = None,
    ) -> torch.Tensor:
        """
        Exact non-adaptive, weight-only SAM delta without materializing full grad_w.

        The additive delta is computed block-by-block over the vocabulary
        dimension, while keeping gradient flow to the post-loss hidden states.
        """
        if chunk_size is None:
            chunk_size = self.sam_vocab_chunk_size
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0.")

        hidden_size = hidden_states.shape[-1]
        vocab_size = grad_logits.shape[-1]

        h_post = hidden_states.reshape(-1, hidden_size)
        h_det = hidden_states.detach().reshape(-1, hidden_size)
        g_det = grad_logits.detach().reshape(-1, vocab_size)

        h_post_out = h_post.to(out_dtype, copy=False)
        with torch.amp.autocast(self.accelerator.device.type, enabled=False):
            h_det_fp32 = h_det.float()
            g_fp32 = g_det.float()

            norm_sq = torch.zeros((), device=g_det.device, dtype=torch.float32)
            for start in range(0, vocab_size, chunk_size):
                end = min(start + chunk_size, vocab_size)
                g_block = g_fp32[:, start:end]
                grad_w_block = g_block.t().matmul(h_det_fp32)
                norm_sq = norm_sq + grad_w_block.square().sum()

            scale = self.sam_rho / norm_sq.sqrt().clamp_min(1e-12)

        delta_logits = torch.empty(
            g_det.shape,
            device=g_det.device,
            dtype=out_dtype,
        )
        for start in range(0, vocab_size, chunk_size):
            end = min(start + chunk_size, vocab_size)
            with torch.amp.autocast(self.accelerator.device.type, enabled=False):
                g_block = g_fp32[:, start:end]
                grad_w_block = g_block.t().matmul(h_det_fp32)
                eps_block_t = (grad_w_block * scale).t().contiguous().to(out_dtype)
            delta_block = h_post_out.matmul(eps_block_t)
            delta_logits[:, start:end] = delta_block

        return delta_logits.view_as(grad_logits)

    def _compute_nonadaptive_delta_logits(
        self,
        hidden_states: torch.Tensor,
        grad_logits: torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Exact non-adaptive, weight-only SAM delta that materializes grad_w.

        This is the simpler dense path and remains the default unless the
        chunked implementation is explicitly enabled.
        """
        hidden_size = hidden_states.shape[-1]
        vocab_size = grad_logits.shape[-1]

        h_post = hidden_states.reshape(-1, hidden_size)
        h_det = hidden_states.detach().reshape(-1, hidden_size)
        g_det = grad_logits.detach().reshape(-1, vocab_size)

        h_post_out = h_post.to(out_dtype, copy=False)
        with torch.amp.autocast(self.accelerator.device.type, enabled=False):
            h_det_fp32 = h_det.float()
            g_fp32 = g_det.float()

            grad_w = g_fp32.t().matmul(h_det_fp32)
            scale = self.sam_rho / grad_w.norm(p=2).clamp_min(1e-12)
            eps_t = (grad_w * scale).t().contiguous().to(out_dtype)
        delta_logits = h_post_out.matmul(eps_t)

        return delta_logits.view_as(grad_logits)

    def _build_perturbed_logits_zero3_nonadaptive_no_gather(
        self,
        hidden_states: torch.Tensor,
        lm_head: nn.Module,
        prepared: dict[str, torch.Tensor | int | None],
        batch: dict[str, torch.Tensor | Any],
        loss_types: list[str],
        aux_loss: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        ZeRO-3 non-adaptive path without GatheredParameters.

        This runs the real lm_head forward once, reuses detached base logits to
        build the SAM direction, and adds an exact chunked delta_logits term.
        """
        head_forward_context = (
            autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        )
        with head_forward_context:
            base_logits = lm_head(hidden_states)

        if not torch.is_tensor(base_logits):
            raise TypeError("lm_head(hidden_states) must return a tensor.")

        clean_logits_pre = base_logits.detach().requires_grad_(True)
        policy_output_pre = self._get_batch_output_from_logits(
            clean_logits_pre,
            prepared,
            aux_loss=aux_loss,
        )
        losses_pre, _, _ = self._compute_weighted_losses(policy_output_pre, batch, loss_types)
        loss_pre = losses_pre.mean()

        (grad_logits,) = torch.autograd.grad(
            loss_pre,
            clean_logits_pre,
            retain_graph=False,
            create_graph=False,
        )

        if self.sam_use_chunked_delta_logits:
            delta_logits = self._compute_nonadaptive_delta_logits_chunked(
                hidden_states=hidden_states,
                grad_logits=grad_logits,
                out_dtype=base_logits.dtype,
                chunk_size=self.sam_vocab_chunk_size,
            )
        else:
            delta_logits = self._compute_nonadaptive_delta_logits(
                hidden_states=hidden_states,
                grad_logits=grad_logits,
                out_dtype=base_logits.dtype,
            )
        perturbed_logits = base_logits + delta_logits
        return perturbed_logits, loss_pre.detach()

    def _compute_sam_perturbation(
        self,
        loss_pre: torch.Tensor,
        clean_logits_pre: torch.Tensor,
        hidden_states_pre: torch.Tensor,
        head_weight: torch.Tensor,
    ) -> torch.Tensor:
        (grad_logits,) = torch.autograd.grad(loss_pre, clean_logits_pre, retain_graph=False, create_graph=False)
        grad_logits_det = grad_logits.detach().reshape(-1, grad_logits.shape[-1])
        hidden_det = hidden_states_pre.reshape(-1, hidden_states_pre.shape[-1]).to(grad_logits_det.dtype, copy=False)
        grad = grad_logits_det.t().matmul(hidden_det)
        if self.sam_adaptive:
            grad = grad * head_weight.detach().abs().to(grad.dtype, copy=False)
        with torch.amp.autocast(self.accelerator.device.type, enabled=False):
            grad_norm = grad.float().norm(p=2).clamp_min(1e-12)
        perturbation = grad * (self.sam_rho / grad_norm).to(grad.dtype)
        return perturbation.to(head_weight.dtype)

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):
        del num_items_in_batch

        if self.is_encoder_decoder:
            raise NotImplementedError("LogitsSAMTrainer currently supports decoder-only models only.")

        compute_loss_context_manager = (
            autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        )

        if "ref_chosen_logps" in inputs and "ref_rejected_logps" in inputs:
            batch = inputs
        else:
            ref_chosen_logps, ref_rejected_logps = self.compute_ref_log_probs(inputs)
            batch = dict(inputs)
            batch["ref_chosen_logps"] = ref_chosen_logps
            batch["ref_rejected_logps"] = ref_rejected_logps

        loss_types = self.loss_type if isinstance(self.loss_type, (list, tuple)) else [self.loss_type]

        prepared = self._prepare_decoder_batch(batch)
        with compute_loss_context_manager:
            hidden_states, lm_head, aux_loss = self._forward_to_hidden_states(model, prepared)

        if _is_zero3(getattr(self, "accelerator", None)):
            if self.sam_adaptive:
                perturbed_logits, loss_pre = self._build_perturbed_logits_zero3(
                    model,
                    hidden_states,
                    lm_head,
                    prepared,
                    batch,
                    loss_types,
                    aux_loss=aux_loss,
                )
            else:
                perturbed_logits, loss_pre = self._build_perturbed_logits_zero3_nonadaptive_no_gather(
                    hidden_states,
                    lm_head,
                    prepared,
                    batch,
                    loss_types,
                    aux_loss=aux_loss,
                )
        else:
            hidden_states_pre = hidden_states.detach()
            head_weight = lm_head.weight
            head_bias = getattr(lm_head, "bias", None)
            clean_logits_pre = self._compute_logits_from_weight_bias(
                hidden_states_pre,
                head_weight.detach(),
                head_bias.detach() if head_bias is not None else None,
            )
            clean_logits_pre = clean_logits_pre.detach().requires_grad_(True)
            policy_output_pre = self._get_batch_output_from_logits(
                clean_logits_pre,
                prepared,
                aux_loss=aux_loss,
            )
            losses_pre, _, _ = self._compute_weighted_losses(policy_output_pre, batch, loss_types)
            loss_pre = losses_pre.mean()

            perturbation = self._compute_sam_perturbation(
                loss_pre,
                clean_logits_pre,
                hidden_states_pre,
                head_weight,
            )
            perturbed_logits = self._compute_logits_from_hidden(hidden_states, lm_head, weight_delta=perturbation)

        policy_output_post = self._get_batch_output_from_logits(
            perturbed_logits,
            prepared,
            aux_loss=aux_loss,
        )
        losses_post, chosen_rewards, rejected_rewards = self._compute_weighted_losses(
            policy_output_post,
            batch,
            loss_types,
        )
        loss_post = losses_post.mean()

        loss = loss_post.to(self.args.device)
        train_eval = "train" if model.training else "eval"

        loss_pre_value = self.accelerator.gather_for_metrics(loss_pre.detach()).mean().item()
        loss_post_value = self.accelerator.gather_for_metrics(loss.detach()).mean().item()
        metrics = {
            "rewards/chosen": self.accelerator.gather_for_metrics(chosen_rewards).mean().item(),
            "rewards/rejected": self.accelerator.gather_for_metrics(rejected_rewards).mean().item(),
            "rewards/accuracies": self.accelerator.gather_for_metrics((chosen_rewards > rejected_rewards).float())
            .mean()
            .item(),
            "rewards/margins": self.accelerator.gather_for_metrics(chosen_rewards - rejected_rewards).mean().item(),
            "logps/chosen": self.accelerator.gather_for_metrics(policy_output_post["chosen_logps"]).detach().mean().item(),
            "logps/rejected": self.accelerator.gather_for_metrics(policy_output_post["rejected_logps"])
            .detach()
            .mean()
            .item(),
            "logits/chosen": self.accelerator.gather_for_metrics(policy_output_post["mean_chosen_logits"])
            .detach()
            .mean()
            .item(),
            "logits/rejected": self.accelerator.gather_for_metrics(policy_output_post["mean_rejected_logits"])
            .detach()
            .mean()
            .item(),
            "sam/loss_pre": loss_pre_value,
            "sam/loss_post": loss_post_value,
            "sam/delta_loss": loss_post_value - loss_pre_value,
        }

        if self.args.rpo_alpha is not None and "nll_loss" in policy_output_post:
            metrics["nll_loss"] = self.accelerator.gather_for_metrics(policy_output_post["nll_loss"]).detach().mean().item()
        if self.aux_loss_enabled and "aux_loss" in policy_output_post:
            metrics["aux_loss"] = self.accelerator.gather_for_metrics(policy_output_post["aux_loss"]).detach().mean().item()

        self.store_metrics(metrics, train_eval=train_eval)

        if return_outputs:
            return loss, metrics
        return loss
