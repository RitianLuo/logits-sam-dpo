import argparse
import logging
import os
import sys

import datasets
import torch
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from alignment import DPOConfig, ScriptArguments, get_dataset, get_model, get_tokenizer
from trl import DPOTrainer, ModelConfig, TrlParser, get_peft_config

logger = logging.getLogger(__name__)


def main(script_args, training_args, model_args, cli_extra=None):
    # Set seed for reproducibility.
    set_seed(training_args.seed)

    # Configure logging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Resume from the last checkpoint when available.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    # Build model, reference model, and tokenizer.
    model = get_model(model_args, training_args)
    ref_model = get_model(model_args, training_args)
    tokenizer = get_tokenizer(model_args, training_args)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(script_args, "ignore_bias_buffers", False):
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # Load dataset and drop chat-formatted messages if present.
    dataset = get_dataset(script_args)
    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    # Select trainer class and trainer-specific kwargs.
    use_sam = bool(getattr(cli_extra, "use_sam_trainer", False))
    if use_sam:
        from sam_trainer import LogitsSAMTrainer as TrainerClass

        extra_kwargs = {
            "sam_rho": float(getattr(cli_extra, "sam_rho", 0.05)),
        }
        logger.info(f"Using LogitsSAMTrainer with {extra_kwargs}")
    else:
        TrainerClass = DPOTrainer
        extra_kwargs = {}
        logger.info("Using vanilla DPOTrainer")

    trainer = TrainerClass(
        model,
        ref_model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        **extra_kwargs,
    )

    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    if training_args.eval_strategy != "no":
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Save and optionally push to hub.
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    # Parse extra CLI options first; pass remaining args to TRL parser.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--use_sam_trainer", action="store_true", help="Use LogitsSAMTrainer instead of DPOTrainer")
    pre.add_argument("--sam_rho", type=float, default=0.05)
    cli_extra, remaining = pre.parse_known_args()

    parser = TrlParser((ScriptArguments, DPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config(remaining)
    main(script_args, training_args, model_args, cli_extra)
