from __future__ import annotations

from typing import Any, Optional, Union
from contextlib import nullcontext, contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autocast  # Generic PyTorch 2.x autocast(device_type)
import torch.distributed as dist
import deepspeed

# TRL / Transformers dependencies
from transformers import PreTrainedModel
from trl.trainer.dpo_trainer import DPOTrainer
from trl.trainer.utils import flush_left, flush_right, selective_log_softmax

def resolve_lm_head(model, accelerator=None):
    if accelerator is not None:
        base = accelerator.unwrap_model(model)
    else:
        try:
            from accelerate.utils import unwrap_model
            base = unwrap_model(model)
        except Exception:
            base = model.module if hasattr(model, "module") else model

    # Support PEFT-wrapped models.
    try:
        from peft import PeftModel
        if isinstance(base, PeftModel):
            base = base.get_base_model()
    except Exception:
        pass

    lm_head = (base.get_output_embeddings() 
               if hasattr(base, "get_output_embeddings") else
               getattr(base, "lm_head", None))
    if lm_head is None:
        raise AttributeError("Output head not found. The model must implement get_output_embeddings() or expose lm_head.")
    return base, lm_head

def _is_zero3(accelerator=None):
    try:
        plug = getattr(getattr(accelerator, "state", None), "deepspeed_plugin", None)
        return getattr(plug, "zero_stage", 0) == 3
    except Exception:
        return False

def _global_l2(x: torch.Tensor) -> torch.Tensor:
    v = x.float().pow(2).sum()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(v, op=dist.ReduceOp.SUM)
    return v.sqrt().clamp_min(1e-12)

@torch.no_grad()
def _zero3_logits_only_from_hidden(
    H_all: torch.Tensor,            # [B, T, hidden] hidden states entering lm_head
    valid_mask: torch.Tensor,       # [B, T] bool mask of supervised positions
    y: torch.Tensor,                # [N] labels at valid positions
    lm_head: nn.Linear,             # output head with weight shape [V, hidden]
    rho: float,
    *,
    fwd_module: nn.Module,          # owning module required by DeepSpeed external params
    use_global_norm: bool = False,  # False means per-rank/per-microbatch norm
):
    # In one gather block: gather W -> compute valid-token gradient -> build e -> produce full perturbed logits.
    with deepspeed.zero.GatheredParameters(
        [lm_head.weight], fwd_module=fwd_module, modifier_rank=0, enabled=True
    ):
        W = lm_head.weight.data
        Wd = W.dtype

        # Compute only on valid positions to save memory.
        HN = H_all[valid_mask]
        logitsN = F.linear(HN.to(Wd, copy=False), W)
        logpN = F.log_softmax(logitsN.float(), dim=-1)
        probsN = logpN.exp()
        idxN = torch.arange(y.shape[0], device=y.device)
        probsN[idxN, y] -= 1.0
        dlogitsN = probsN / max(int(y.shape[0]), 1)

        # g = dL/dW = dlogits^T @ H
        g = dlogitsN.t().matmul(HN.float())

        # Local norm by default, optionally global norm across ranks.
        n = _global_l2(g) if use_global_norm else g.float().norm(p=2).clamp_min(1e-12)

        e = (rho / n).to(Wd) * g.to(Wd)

        # Build full perturbed logits in one linear pass.
        logits2_full = F.linear(H_all.to(Wd, copy=False), W + e)
        return logits2_full, e

@contextmanager
def _perturb_(param: torch.Tensor, delta: torch.Tensor):
    with torch.no_grad():
        param.add_(delta)
    try:
        yield
    finally:
        with torch.no_grad():
            param.sub_(delta)
            
class LogitsSAMTrainer(DPOTrainer):
    """
    DPO trainer with logits-only SAM updates.

    It reuses TRL's concatenated forward/log-prob path and applies
    SAM perturbation on lm_head without a second full model forward.
    """
    def __init__(self, *args, sam_rho: float = 0.05, sam_adaptive: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self.sam_rho = sam_rho
        self.sam_adaptive = sam_adaptive
        
    def concatenated_forward_with_logits(
        self,
        precomputed_logits: Optional[torch.Tensor],
        model: nn.Module,
        batch: dict[str, Union[list, torch.LongTensor]],
        is_ref_model: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Runs the given model on the given batch of inputs, concatenating the chosen and rejected inputs together.
    
        We do this to avoid doing two forward passes, because it's faster for FSDP.
    
        Args:
            model:
                Model to run the forward pass on.
            batch:
                Batch of input data.
            is_ref_model:
                Whether this method is being called for the reference model. If `True`, length desensitization is not
                applied.
        """
        num_examples = batch["prompt_input_ids"].shape[0]
    
        concatenated_batch = self.concatenated_inputs(batch, padding_value=self.padding_value)
    
        model_kwargs = {"use_cache": False}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True
    
        # Add the pixel values and attention masks for vision models
        if "pixel_values" in concatenated_batch:
            model_kwargs["pixel_values"] = concatenated_batch["pixel_values"]
        if "pixel_attention_mask" in concatenated_batch:
            model_kwargs["pixel_attention_mask"] = concatenated_batch["pixel_attention_mask"]
        if "image_sizes" in concatenated_batch:
            model_kwargs["image_sizes"] = concatenated_batch["image_sizes"]
    
        prompt_input_ids = concatenated_batch["prompt_input_ids"]
        prompt_attention_mask = concatenated_batch["prompt_attention_mask"]
        completion_input_ids = concatenated_batch["completion_input_ids"]
        completion_attention_mask = concatenated_batch["completion_attention_mask"]
        
        if self.is_encoder_decoder:
            labels = completion_input_ids
            labels[completion_attention_mask == 0] = self.label_pad_token_id
            outputs = model(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                labels=labels,  # we need the labels for the logits to be returned
                **model_kwargs,
            )
            logits = outputs.logits
            loss_mask = completion_attention_mask.bool()
        else:
            # Concatenate the prompt and completion inputs
            input_ids = torch.cat((prompt_input_ids, completion_input_ids), dim=1)
            attention_mask = torch.cat((prompt_attention_mask, completion_attention_mask), dim=1)
            # Mask the prompt but not the completion for the loss
            loss_mask = torch.cat(
                (torch.zeros_like(prompt_attention_mask), completion_attention_mask),
                dim=1,
            )
    
            # Flush and truncate
            if self.max_length is not None and self.max_length < attention_mask.size(1):
                if self.truncation_mode == "keep_start":
                    # Flush left to reduce the memory usage
                    # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
                    #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
                    attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
                    attention_mask = attention_mask[:, : self.max_length]
                    input_ids = input_ids[:, : self.max_length]
                    loss_mask = loss_mask[:, : self.max_length]
                elif self.truncation_mode == "keep_end":
                    # Flush right before truncating left, then flush left
                    # [[0, 0, x, x, x, x],  ->  [[0, 0, x, x],
                    #  [0, x, x, x, 0, 0]]       [0, x, x, x]]
                    attention_mask, input_ids, loss_mask = flush_right(attention_mask, input_ids, loss_mask)
                    input_ids = input_ids[:, -self.max_length :]
                    attention_mask = attention_mask[:, -self.max_length :]
                    loss_mask = loss_mask[:, -self.max_length :]
                    attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
                else:
                    raise ValueError(
                        f"Unknown truncation mode: '{self.truncation_mode}'. Should be one of ['keep_end', "
                        "'keep_start']."
                    )
            else:
                # Flush left to reduce the memory usage
                # [[0, 0, x, x, x, x],  ->  [[x, x, x, x],
                #  [0, x, x, x, 0, 0]]       [x, x, x, 0]]
                attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
    
            if self.use_logits_to_keep:
                # Compute logits_to_keep based on loss_mask pattern:
                # [[0, 0, 0, x, x, x, x],
                #  [0, 0, 0, x, x, x, 0]]
                #         ^ start computing logits from here ([:, -(7-3+1):])
                first_compute_index = loss_mask.nonzero(as_tuple=True)[1].min()
                logits_to_keep = (loss_mask.shape[1] - first_compute_index).item() + 1  # +1 for the first label
                model_kwargs["logits_to_keep"] = logits_to_keep
    
            model_kwargs["output_hidden_states"] = True
    
            if self.padding_free:
                # Flatten the input_ids, position_ids, and loss_mask
                # input_ids = [[a, b, c, 0], ->     input_ids = [[a, b, c, d, e, f, g]]
                #              [d, e, f, g]]     position_ids = [[0, 1, 2, 0, 1, 2, 3]]
                input_ids = input_ids[attention_mask.bool()].unsqueeze(0)
                loss_mask = loss_mask[attention_mask.bool()].unsqueeze(0)
                position_ids = attention_mask.cumsum(1)[attention_mask.bool()].unsqueeze(0) - 1
                model_kwargs["position_ids"] = position_ids
            else:
                model_kwargs["attention_mask"] = attention_mask

            # Only call model when precomputed logits are not provided.
            if precomputed_logits is None:
                model_kwargs["output_hidden_states"] = True
                outputs = model(input_ids, **model_kwargs)
                logits = outputs.logits
            else:
                logits = precomputed_logits
            labels = torch.roll(input_ids, shifts=-1, dims=1)
            loss_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()
    
            if self.use_logits_to_keep:
                # Align labels with logits
                # logits:    -,  -, [x2, x3, x4, x5, x6]
                #                     ^ --------- ^       after logits[:, :-1, :]
                # labels:   [y0, y1, y2, y3, y4, y5, y6]
                #                         ^ --------- ^   with logits_to_keep=4, [:, -4:]
                # loss_mask: [0,  0,  0,  1,  1,  1,  1]
                labels = labels[:, -logits_to_keep:]
                loss_mask = loss_mask[:, -logits_to_keep:]
        if logits.shape[:2] != labels.shape[:2]:
            # for LLaVA, the returned logits include the image tokens (placed before the text tokens)
            seq_len = labels.shape[1]
            logits = logits[:, -seq_len:]
    
        # Compute the log probabilities of the labels
        labels[~loss_mask] = 0  # dummy token; we'll ignore the losses on these tokens later
        per_token_logps = selective_log_softmax(logits, labels)
        per_token_logps[~loss_mask] = 0
        per_token_logps = torch.roll(per_token_logps, shifts=1, dims=1)
    
        if self.padding_free:
            # Unflatten the per_token_logps (shape: [1, sum_seq_len] -> [batch_size, seq_len])
            batch_size, seq_len = attention_mask.shape
            per_token_logps_ = torch.zeros(
                batch_size, seq_len, device=logits.device, dtype=logits.dtype
            )
            per_token_logps_[attention_mask.bool()] = per_token_logps
            per_token_logps = per_token_logps_
    
        all_logps = per_token_logps[:, 1:].sum(-1)
    
        output = {}
    
        if self.use_weighting:
            with torch.no_grad():
                # Eq (2) of the WPO paper: https://huggingface.co/papers/2406.11827
                logprobs = F.log_softmax(logits, dim=-1)
                weights_adjustment_factor = torch.logsumexp(2 * logprobs, dim=-1)  # same as sum(probs**2) in log space
                per_token_logps_adjusted = per_token_logps - weights_adjustment_factor
                all_weights = (per_token_logps_adjusted * loss_mask).sum(-1) / loss_mask.sum(-1)
                chosen_weights = all_weights[:num_examples]
                rejected_weights = all_weights[num_examples:]
                output["policy_weights"] = torch.clamp(torch.exp(chosen_weights + rejected_weights), max=1)
    
        if self.args.rpo_alpha is not None or "sft" in self.loss_type:
            # Only use the chosen logits for the RPO loss or SFT loss
            chosen_logits = logits[:num_examples, :-1] if not self.is_encoder_decoder else logits[:num_examples]
            chosen_labels = labels[:num_examples, :-1] if not self.is_encoder_decoder else labels[:num_examples]
    
            # Compute the log probabilities of the labels
            output["nll_loss"] = F.cross_entropy(
                torch.flatten(chosen_logits, end_dim=1), torch.flatten(chosen_labels, end_dim=1), ignore_index=0
            )
    
        if "ipo" in self.loss_type:
            all_logps = all_logps / loss_mask.sum(-1)
    
        if self.args.ld_alpha is not None and not is_ref_model:
            # Compute response lengths based on loss_mask
            completion_lengths = loss_mask.sum(dim=1)
    
            chosen_lengths = completion_lengths[:num_examples]
            rejected_lengths = completion_lengths[num_examples:]
            public_lengths = torch.min(chosen_lengths, rejected_lengths)  # l_p in the paper
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
    
        # Compute the mean logits
        if self.padding_free:
            # position_ids contains a sequence of range identifiers (e.g., [[0, 1, 2, 0, 1, 2, 3, ...]]).
            # There are 2*num_examples ranges in total: the first half corresponds to the chosen tokens,
            # and the second half to the rejected tokens.
            # To find the start of the rejected tokens, we look for the num_examples+1-th zero in pos_id.
            split_idx = (position_ids == 0).nonzero(as_tuple=True)[1][num_examples]
            mean_chosen_logits = logits[0, :split_idx][loss_mask[0, :split_idx]].mean()
            mean_rejected_logits = logits[0, split_idx:][loss_mask[0, split_idx:]].mean()
        else:
            mean_chosen_logits = logits[:num_examples][loss_mask[:num_examples]].mean()
            mean_rejected_logits = logits[num_examples:][loss_mask[num_examples:]].mean()
    
        output["mean_chosen_logits"] = mean_chosen_logits
        output["mean_rejected_logits"] = mean_rejected_logits
    
        if self.aux_loss_enabled and 'outputs' in locals():
            output["aux_loss"] = outputs.aux_loss
        return output
        
    def build_perturbed_logits_zero3(
        self,
        model: nn.Module,
        inputs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Build perturbed logits for ZeRO-3 with a logits-only SAM path.

        This method mirrors TRL's concatenation/flush/truncation behavior,
        runs one normal forward to capture hidden states entering lm_head,
        and recomputes only lm_head for perturbed logits.

        Returns:
            logits_perturbed: [B, T_used, V]
            loss_proxy: scalar tensor used to build the SAM perturbation
        """
        if self.is_encoder_decoder:
            raise NotImplementedError("Logits-only SAM (ZeRO-3) currently supports decoder-only models.")

        # 1) Build concatenated batch exactly like TRL.
        num_examples = inputs["prompt_input_ids"].shape[0]
        concatenated_batch = self.concatenated_inputs(inputs, padding_value=self.padding_value)

        model_kwargs = {"use_cache": False}
        if self.aux_loss_enabled:
            model_kwargs["output_router_logits"] = True
        for k in ("pixel_values", "pixel_attention_mask", "image_sizes"):
            if k in concatenated_batch:
                model_kwargs[k] = concatenated_batch[k]

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
                input_ids = input_ids[:, -self.max_length:]
                attention_mask = attention_mask[:, -self.max_length:]
                loss_mask = loss_mask[:, -self.max_length:]
                attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)
            else:
                raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")
        else:
            attention_mask, input_ids, loss_mask = flush_left(attention_mask, input_ids, loss_mask)

        if self.use_logits_to_keep:
            first_compute_index = loss_mask.nonzero(as_tuple=True)[1].min()
            logits_to_keep = (loss_mask.shape[1] - first_compute_index).item() + 1
            model_kwargs["logits_to_keep"] = logits_to_keep

        model_kwargs["output_hidden_states"] = True

        if self.padding_free:
            input_ids = input_ids[attention_mask.bool()].unsqueeze(0)
            loss_mask = loss_mask[attention_mask.bool()].unsqueeze(0)
            position_ids = attention_mask.cumsum(1)[attention_mask.bool()].unsqueeze(0) - 1
            model_kwargs["position_ids"] = position_ids
        else:
            model_kwargs["attention_mask"] = attention_mask

        # 2) First forward pass: capture hidden states entering lm_head.
        caps = {}
        _, lm_head = resolve_lm_head(model, accelerator=self.accelerator)

        def _pre_head(_m, inp):
            caps["H_all"] = inp[0]

        h = lm_head.register_forward_pre_hook(_pre_head)
        try:
            outputs = model(input_ids, **model_kwargs)
        finally:
            h.remove()

        logits = outputs.logits

        labels_full = torch.roll(input_ids, shifts=-1, dims=1)
        valid_mask = torch.roll(loss_mask, shifts=-1, dims=1).bool()

        if self.use_logits_to_keep:
            labels_full = labels_full[:, -logits_to_keep:]
            valid_mask = valid_mask[:, -logits_to_keep:]

        # Match TRL behavior: set invalid labels to dummy token id 0.
        labels_full[~valid_mask] = 0

        if logits.shape[:2] != labels_full.shape[:2]:
            seq_len = labels_full.shape[1]
            logits = logits[:, -seq_len:]

        H_all = caps["H_all"]
        H_all = H_all if H_all.is_contiguous() else H_all.contiguous()

        # 3) Build compact valid-token views.
        B, T = labels_full.shape
        row_ids = torch.arange(B, device=labels_full.device).unsqueeze(1).expand(B, T)
        idxN = row_ids[valid_mask].contiguous()
        yN = labels_full[valid_mask].contiguous()
        HN = H_all[valid_mask]

        if "ref_chosen_logps" in inputs and "ref_rejected_logps" in inputs:
            ref_ch, ref_rj = inputs["ref_chosen_logps"], inputs["ref_rejected_logps"]
        else:
            ref_ch, ref_rj = self.compute_ref_log_probs(inputs)

        # 4) ZeRO-3 gather once and complete logitsN -> dL/dlogitsN -> e -> logits_perturbed.
        with deepspeed.zero.GatheredParameters([lm_head.weight], fwd_module=model, modifier_rank=0, enabled=True):
            # Keep grad path for perturbed forward while keeping e first-order.
            W_param = lm_head.weight
            W_const = W_param.detach()
            Wd = W_param.dtype

            with self.accelerator.autocast():
                logitsN = F.linear(HN.to(Wd, copy=False), W_const)
            logitsN = logitsN.detach().requires_grad_(True)

            tok_logp = selective_log_softmax(logitsN, yN)
            logps_per_sample = torch.zeros(B, device=tok_logp.device, dtype=tok_logp.dtype)
            logps_per_sample.index_add_(0, idxN, tok_logp)
            chosen_logps_proxy = logps_per_sample[:num_examples]
            rejected_logps_proxy = logps_per_sample[num_examples:]

            loss_types = self.loss_type if isinstance(self.loss_type, (list, tuple)) else [self.loss_type]
            losses_for_grad = 0.0
            for i, lt in enumerate(loss_types):
                _losses, _, _ = self.dpo_loss(
                    chosen_logps_proxy,
                    rejected_logps_proxy,
                    ref_ch,
                    ref_rj,
                    lt,
                    {},
                )
                w = self.loss_weights[i] if self.loss_weights else 1.0
                losses_for_grad = losses_for_grad + _losses * w

            if self.args.rpo_alpha is not None:
                is_chosen = idxN < num_examples
                if is_chosen.any():
                    logits_ch = logitsN[is_chosen]
                    y_ch = yN[is_chosen]
                    nll_ch = F.cross_entropy(logits_ch, y_ch, reduction="mean", ignore_index=0)
                    losses_for_grad = losses_for_grad + self.args.rpo_alpha * nll_ch

            (dL_dlogitsN,) = torch.autograd.grad(
                losses_for_grad.mean(), logitsN, retain_graph=False, create_graph=False
            )
            with torch.amp.autocast("cuda", enabled=False):
                dL = dL_dlogitsN.float()
                H32 = HN.detach().float()  # keep e first-order (no second-order graph)
                g32 = dL.t().matmul(H32)
                if getattr(self, "sam_adaptive", False):
                    g32 = g32 * W_const.abs().float()
                n32 = g32.norm(p=2).clamp_min(1e-12)
                e32 = g32 * (self.sam_rho / n32)
            e = e32.to(Wd).detach()

            with self.accelerator.autocast():
                # Keep dependency on lm_head.weight so perturbed loss updates the head.
                logits_perturbed = F.linear(H_all.to(Wd, copy=False), W_param + e)

        if logits_perturbed.shape[:2] != labels_full.shape[:2]:
            seq_len = labels_full.shape[1]
            logits_perturbed = logits_perturbed[:, -seq_len:]

        return logits_perturbed, losses_for_grad.detach()

    @torch.no_grad()
    def _get_ref_logps(self, inputs):
        # Reuse TRL ref-logps flow: prefer cached values, otherwise run ref_model.
        if self.precompute_ref_log_probs:
            return inputs["ref_chosen_logps"], inputs["ref_rejected_logps"]

        ref_outputs = self.concatenated_forward(self.ref_model, inputs)
        return ref_outputs["chosen_logps"], ref_outputs["rejected_logps"]

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: dict[str, Union[torch.Tensor, Any]],
        return_outputs: bool = False,
        num_items_in_batch=None,
    ):

        if self.is_encoder_decoder:
            raise NotImplementedError("Logits-only SAM currently supports decoder-only models.")

        # Keep precision strategy aligned with the base trainer behavior.
        compute_loss_context_manager = (
            autocast(self.accelerator.device.type) if self._peft_has_been_casted_to_bf16 else nullcontext()
        )

        # Compute reference logps once and cache them in batch.
        if "ref_chosen_logps" in inputs and "ref_rejected_logps" in inputs:
            batch = inputs
        else:
            ref_ch, ref_rj = self.compute_ref_log_probs(inputs)
            batch = dict(inputs)
            batch["ref_chosen_logps"] = ref_ch
            batch["ref_rejected_logps"] = ref_rj

        loss_types = self.loss_type if isinstance(self.loss_type, (list, tuple)) else [self.loss_type]

        if _is_zero3(getattr(self, "accelerator", None)):
            # ZeRO-3 path: build perturbed logits with logits-only SAM.
            logits_perturbed, loss_pre = self.build_perturbed_logits_zero3(model, batch)
        else:
            # Non-ZeRO-3 path: capture hidden states entering lm_head.
            _, lm_head = resolve_lm_head(model, accelerator=self.accelerator)
            head_w: torch.Tensor = lm_head.weight
            caps = {}

            def _pre_head(_m, inp):
                caps["H"] = inp[0]

            h = lm_head.register_forward_pre_hook(_pre_head)
            try:
                with compute_loss_context_manager:
                    po = self.concatenated_forward(model, batch)
            finally:
                h.remove()

            H = caps.get("H", None)
            H = H if H.is_contiguous() else H.contiguous()
            if H is None:
                raise RuntimeError("Failed to capture hidden states entering lm_head.")

            losses_pre = 0.0
            for idx, lt in enumerate(loss_types):
                _losses, _, _ = self.dpo_loss(
                    po["chosen_logps"],
                    po["rejected_logps"],
                    batch["ref_chosen_logps"],
                    batch["ref_rejected_logps"],
                    lt,
                    po,
                )
                w = self.loss_weights[idx] if self.loss_weights else 1.0
                losses_pre += _losses * w

            if self.args.rpo_alpha is not None and "nll_loss" in po:
                losses_pre = losses_pre + self.args.rpo_alpha * po["nll_loss"]
            if self.use_weighting and "policy_weights" in po:
                losses_pre = losses_pre * po["policy_weights"]
            if self.aux_loss_enabled and "aux_loss" in po:
                losses_pre = losses_pre + self.aux_loss_coef * po["aux_loss"]

            loss_pre = losses_pre.mean()

            # Build SAM perturbation from lm_head gradient only.
            g = torch.autograd.grad(loss_pre, head_w, retain_graph=False, allow_unused=False)[0]
            if g is None:
                raise RuntimeError("autograd.grad returned None for lm_head.weight; check requires_grad / graph.")

            if getattr(self, "sam_adaptive", False):
                g = g * head_w.detach().abs()
            with torch.amp.autocast("cuda", enabled=False):
                g32 = g.detach().float()
                n32 = g32.norm(p=2).clamp_min(1e-12)
                e32 = g32 * (self.sam_rho / n32)

            e = e32.to(head_w.dtype)
            with self.accelerator.autocast():
                logits_perturbed = torch.nn.functional.linear(H.to(head_w.dtype, copy=False), head_w + e)

        with compute_loss_context_manager:
            # Reuse perturbed logits directly to avoid another full model forward.
            po2 = self.concatenated_forward_with_logits(logits_perturbed, model, batch)

            losses_post = 0.0
            chosen_rewards = 0.0
            rejected_rewards = 0.0
            for idx, lt in enumerate(loss_types):
                _losses, _cr, _rr = self.dpo_loss(
                    po2["chosen_logps"],
                    po2["rejected_logps"],
                    batch["ref_chosen_logps"],
                    batch["ref_rejected_logps"],
                    lt,
                    po2,
                )
                w = self.loss_weights[idx] if self.loss_weights else 1.0
                losses_post += _losses * w
                chosen_rewards += _cr * w
                rejected_rewards += _rr * w

            if self.args.rpo_alpha is not None and "nll_loss" in po2:
                losses_post = losses_post + self.args.rpo_alpha * po2["nll_loss"]
            if self.use_weighting and "policy_weights" in po2:
                losses_post = losses_post * po2["policy_weights"]
            if self.aux_loss_enabled and "aux_loss" in po2:
                losses_post = losses_post + self.aux_loss_coef * po2["aux_loss"]

        loss_post = losses_post.mean()

        # Match base trainer behavior: move loss to main device and log metrics.
        loss = loss_post.to(self.args.device)

        metrics = {
            "rewards/chosen": self.accelerator.gather_for_metrics(chosen_rewards).mean().item(),
            "rewards/rejected": self.accelerator.gather_for_metrics(rejected_rewards).mean().item(),
            "rewards/accuracies": self.accelerator.gather_for_metrics((chosen_rewards > rejected_rewards).float()).mean().item(),
            "rewards/margins": self.accelerator.gather_for_metrics(chosen_rewards - rejected_rewards).mean().item(),
            "logps/chosen": self.accelerator.gather_for_metrics(po2["chosen_logps"]).detach().mean().item(),
            "logps/rejected": self.accelerator.gather_for_metrics(po2["rejected_logps"]).detach().mean().item(),
            "logits/chosen": self.accelerator.gather_for_metrics(po2["mean_chosen_logits"]).detach().mean().item(),
            "logits/rejected": self.accelerator.gather_for_metrics(po2["mean_rejected_logits"]).detach().mean().item(),
            "sam/loss_pre": loss_pre.detach().mean().item(),
            "sam/loss_post": loss.detach().mean().item(),
        }
        if self.args.rpo_alpha is not None and "nll_loss" in po2:
            metrics["nll_loss"] = self.accelerator.gather_for_metrics(po2["nll_loss"]).detach().mean().item()
        if self.aux_loss_enabled and "aux_loss" in po2:
            metrics["aux_loss"] = self.accelerator.gather_for_metrics(po2["aux_loss"]).detach().mean().item()

        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return loss, metrics
        return loss
    
