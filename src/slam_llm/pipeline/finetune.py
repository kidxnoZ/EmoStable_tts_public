# os
import os
import fire
import random
import importlib
from copy import deepcopy

# nn
import torch
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

# opt
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.distributed.fsdp.fully_sharded_data_parallel import CPUOffload
from slam_llm.policies import AnyPrecisionAdamW, apply_fsdp_checkpointing

# util
from slam_llm.utils import fsdp_auto_wrap_policy
from slam_llm.utils.config_utils import get_dataloader_kwargs

from slam_llm.utils.dataset_utils import get_preprocessed_dataset
from slam_llm.data.concatenator import ConcatDataset

from slam_llm.utils.model_utils import get_custom_model_factory
from slam_llm.utils.train_utils import (
    train,
    freeze_transformer_layers,
    setup,
    setup_environ_flags,
    clear_gpu_cache,
    get_policies
)

import sys
import logging
import wandb

import hydra
from omegaconf import DictConfig, ListConfig, OmegaConf
from pathlib import Path

def _str2bool(value):
    value = str(value).strip().lower()
    return value in {"1", "true", "t", "yes", "y", "on"}


def _debug_train_sanity_enabled(train_config):
    return bool(train_config.get("debug_train_sanity", False)) or _str2bool(
        os.environ.get("DEBUG_TRAIN_SANITY", "0")
    )


def _apply_tiny_overfit_if_needed(dataset_train, train_config, logger):
    tiny_mode = bool(train_config.get("tiny_overfit_mode", False))
    if not tiny_mode:
        return {"enabled": False}

    if not hasattr(dataset_train, "data_list"):
        logger.warning("[SANITY] tiny_overfit_mode enabled but dataset has no data_list; skip tiny-overfit.")
        return {"enabled": False, "reason": "dataset_has_no_data_list"}

    if len(dataset_train.data_list) == 0:
        logger.warning("[SANITY] tiny_overfit_mode enabled but training dataset is empty.")
        return {"enabled": False, "reason": "empty_dataset"}

    keep_n = max(1, int(train_config.get("tiny_overfit_num_samples", 4)))
    keep_n = min(keep_n, len(dataset_train.data_list))
    repeat_factor = max(1, int(train_config.get("tiny_overfit_repeat", 64)))
    kept = list(dataset_train.data_list[:keep_n])
    dataset_train.data_list = kept * repeat_factor
    logger.warning(
        "[SANITY] tiny_overfit_mode enabled: keeping first %d samples and repeating x%d (train_len=%d).",
        keep_n,
        repeat_factor,
        len(dataset_train.data_list),
    )
    return {
        "enabled": True,
        "kept_samples": keep_n,
        "repeat_factor": repeat_factor,
        "train_len": len(dataset_train.data_list),
    }


@hydra.main(config_name=None, version_base=None)
def main_hydra(cfg: DictConfig):
    def to_plain_list(cfg_item):
        if isinstance(cfg_item, ListConfig):
            return OmegaConf.to_container(cfg_item, resolve=True)
        elif isinstance(cfg_item, DictConfig):
            return {k: to_plain_list(v) for k, v in cfg_item.items()}
        else:
            return cfg_item
    
    # kwargs = to_plain_list(cfg)
    kwargs = cfg
    log_level = getattr(logging, kwargs.get("log_level", "INFO").upper())
    
    logging.basicConfig(level=log_level)
    
    if kwargs.get("debug", False):
        import pdb;
        pdb.set_trace()
        
    main(kwargs)


def main(kwargs: DictConfig):
    train_config, fsdp_config, model_config, log_config, dataset_config = kwargs.train_config, \
                                                                          kwargs.fsdp_config, \
                                                                          kwargs.model_config, \
                                                                          kwargs.log_config, \
                                                                          kwargs.dataset_config
    
    fsdp_config.use_fp16 = train_config.use_fp16
    OmegaConf.set_struct(kwargs,False)
    del kwargs["train_config"]
    del kwargs["fsdp_config"]
    del kwargs["model_config"]
    del kwargs["log_config"]
    del kwargs["dataset_config"]
    OmegaConf.set_struct(kwargs,True)
    
    # Set log
    if not os.path.exists(os.path.dirname(log_config.log_file)):
        os.makedirs(os.path.dirname(log_config.log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        filemode='w'
    )

    logger = logging.getLogger()  
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(filename=log_config.log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler.setFormatter(file_formatter)

    logger.handlers[0].setLevel(logging.INFO)
    console_formatter = logging.Formatter('[%(asctime)s][%(name)s][%(levelname)s] - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    logger.handlers[0].setFormatter(console_formatter) 

    logger.addHandler(file_handler)


    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)
    random.seed(train_config.seed)

    if train_config.enable_fsdp or train_config.enable_ddp:
        setup()
        # torchrun specific
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        logger.info(f"local_rank: {local_rank}, rank: {rank}, world_size: {world_size}")

    if torch.distributed.is_initialized():
        torch.cuda.set_device(local_rank)
        clear_gpu_cache(local_rank)
        setup_environ_flags(rank)

    if not (train_config.enable_fsdp or train_config.enable_ddp) or rank == 0:
        logger.info("train_config: {}".format(train_config))
        logger.info("fsdp_config: {}".format(fsdp_config))
        logger.info("model_config: {}".format(model_config))
        logger.info("log_config: {}".format(log_config))
    debug_train_sanity = _debug_train_sanity_enabled(train_config)
    if not (train_config.enable_fsdp or train_config.enable_ddp) or rank == 0:
        logger.info(f"[SANITY] debug_train_sanity={debug_train_sanity}")
        if debug_train_sanity:
            logger.warning(
                "[SANITY] Debug instrumentation is enabled; extra diagnostics will be logged."
            )
        logger.info(f"[CKPT] policy_init_ckpt_path={kwargs.get('ckpt_path', None)}")

    # Set wandb
    if not (train_config.enable_fsdp or train_config.enable_ddp) or rank == 0:
        if log_config.use_wandb:
            if not os.path.exists(log_config.wandb_dir):
                os.makedirs(log_config.wandb_dir, exist_ok=True)
            wandb_config={"train_config": train_config, "fsdp_config": fsdp_config, "model_config": model_config, "log_config": log_config}
            wandb.init(dir=log_config.wandb_dir, entity=log_config.wandb_entity_name, project=log_config.wandb_project_name,name=log_config.wandb_exp_name ,config=wandb_config)


    model_factory = get_custom_model_factory(model_config, logger)
    model, tokenizer = model_factory(train_config, model_config, **kwargs)
    reference_model = None
    reference_model_path = None
    if train_config.get("use_dpo", False):
        reference_model_path = train_config.get("reference_model_path", None)
        if reference_model_path in (None, "", "None"):
            raise ValueError("train_config.reference_model_path must be set when train_config.use_dpo=true")
        ref_train_config = deepcopy(train_config)
        ref_train_config.use_peft = False
        ref_train_config.enable_fsdp = False
        ref_train_config.enable_ddp = False
        ref_train_config.enable_deepspeed = False
        ref_train_config.quantization = False
        ref_kwargs = {k: kwargs.get(k) for k in kwargs.keys()}
        ref_kwargs["ckpt_path"] = reference_model_path
        ref_kwargs["peft_ckpt"] = None
        reference_model, _ = model_factory(ref_train_config, model_config, **ref_kwargs)
        for _, param in reference_model.named_parameters():
            param.requires_grad = False
        reference_model.eval()
        logger.info(f"Loaded frozen reference model from: {reference_model_path}")
        if debug_train_sanity:
            logger.info(f"[CKPT] reference_model_path={reference_model_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    
    # Convert the model to bfloat16 if fsdp and pure_bf16 is enabled
    if (train_config.enable_fsdp or train_config.enable_ddp) and fsdp_config.pure_bf16:
        model.to(torch.bfloat16)

    #setting up FSDP if enable_fsdp is enabled
    if train_config.enable_fsdp:
        if not train_config.use_peft and train_config.freeze_layers:

            freeze_transformer_layers(train_config.num_freeze_layers)
        from torch.distributed.fsdp import ShardingStrategy
        fsdp_config.sharding_strategy = getattr(ShardingStrategy, fsdp_config.sharding_strategy)
        mixed_precision_policy, wrapping_policy = get_policies(fsdp_config, rank)
        my_auto_wrapping_policy = fsdp_auto_wrap_policy(model, LlamaDecoderLayer)

        model = FSDP(
            model,
            auto_wrap_policy= my_auto_wrapping_policy, #(FIX:MZY): Using my_auto_wrapping_policy whether peft or not. This will avoid model shard type check error of requires_grad mismatching.
            cpu_offload=CPUOffload(offload_params=True) if fsdp_config.fsdp_cpu_offload else None,
            mixed_precision=mixed_precision_policy if not fsdp_config.pure_bf16 else None,
            sharding_strategy=fsdp_config.sharding_strategy,
            device_id=torch.cuda.current_device(),
            limit_all_gathers=True,
            sync_module_states=train_config.low_cpu_fsdp,
            param_init_fn=lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
            if train_config.low_cpu_fsdp and rank != 0 else None,
        )
        if fsdp_config.fsdp_activation_checkpointing:
            apply_fsdp_checkpointing(model)
    elif train_config.enable_ddp:
        model = model.cuda(local_rank)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=train_config.get("find_unused_parameters", False))
    elif not train_config.quantization:
        model.to(device)

    if reference_model is not None:
        if train_config.enable_fsdp or train_config.enable_ddp:
            reference_model = reference_model.to(local_rank)
        else:
            reference_model = reference_model.to(device)
        reference_model.eval()

    # dataset_config = generate_dataset_config(train_config, kwargs)
    logger.info("dataset_config: {}".format(dataset_config))
    if not (train_config.enable_fsdp or train_config.enable_ddp) or rank == 0:
        if log_config.use_wandb:
            wandb.config.update({"dataset_config": dataset_config})
    
    # Load and preprocess the dataset for training and validation
    dataset_train_config = dataset_config
    dataset_val_config = dataset_config
    if train_config.get("use_dpo", False):
        if train_config.batching_strategy == "packing":
            raise ValueError("DPO mode does not support packing batching_strategy; please use custom/padding")
        dpo_pair_jsonl = train_config.get("dpo_pair_jsonl", None)
        if dpo_pair_jsonl in (None, "", "None"):
            raise ValueError("train_config.dpo_pair_jsonl must be set when train_config.use_dpo=true")
        repo_root = Path(__file__).resolve().parents[3]
        dpo_dataset_file = (repo_root / "examples" / "tts" / "speech_dataset_tts_dpo.py").as_posix()
        dataset_train_config = OmegaConf.create(OmegaConf.to_container(dataset_config, resolve=True))
        dataset_train_config.file = f"{dpo_dataset_file}:get_speech_dataset_dpo"
        dataset_train_config.train_data_path = dpo_pair_jsonl
        dpo_excluded_emotions = train_config.get(
            "dpo_excluded_emotions",
            ["fearful", "surprised", "disgusted"],
        )
        dataset_train_config.dpo_excluded_emotions = dpo_excluded_emotions
        logger.info(f"DPO mode enabled: train dataset switched to pair file {dpo_pair_jsonl}")
        logger.info(f"DPO mode enabled: excluded emotions={list(dpo_excluded_emotions)}")
        if train_config.run_validation:
            dpo_val_pair_jsonl = train_config.get("dpo_val_pair_jsonl", None)
            if dpo_val_pair_jsonl in (None, "", "None"):
                dpo_val_pair_jsonl = dpo_pair_jsonl
            dataset_val_config = OmegaConf.create(OmegaConf.to_container(dataset_config, resolve=True))
            dataset_val_config.file = f"{dpo_dataset_file}:get_speech_dataset_dpo"
            dataset_val_config.val_data_path = dpo_val_pair_jsonl
            dataset_val_config.dpo_excluded_emotions = dpo_excluded_emotions
            logger.info(f"DPO mode enabled: val dataset switched to pair file {dpo_val_pair_jsonl}")
    if debug_train_sanity:
        train_data_path = dataset_train_config.get("train_data_path", None)
        val_data_path = dataset_val_config.get("val_data_path", None)
        logger.info(f"[SANITY] dataset_train_path={train_data_path}")
        logger.info(f"[SANITY] dataset_val_path={val_data_path}")
        if str(dataset_train_config.get("inference_mode", False)).lower() in {"1", "true"}:
            logger.warning("[DIAGNOSIS] [BATCH] dataset_train_config.inference_mode=True may break training labels.")
        for split_name, data_path in [("train", train_data_path), ("val", val_data_path)]:
            if data_path is None:
                continue
            if Path(str(data_path)).name.lower().startswith("test"):
                logger.warning(
                    f"[DIAGNOSIS] [BATCH] {split_name}_data_path points to a test file: {data_path}"
                )

    dataset_train = get_preprocessed_dataset(
        tokenizer,
        dataset_train_config,
        split="train",
    )
    if not (train_config.enable_fsdp or train_config.enable_ddp) or rank == 0:
        logger.info(f"--> Training Set Length = {len(dataset_train)}")
    dataset_val = get_preprocessed_dataset(
        tokenizer,
        dataset_val_config,
        split="val",
    )
    if not (train_config.enable_fsdp or train_config.enable_ddp) or rank == 0:
        logger.info(f"--> Validation Set Length = {len(dataset_val)}")
    if train_config.batching_strategy == "packing":
        dataset_train = ConcatDataset(dataset_train, chunk_size=train_config.context_length)

    tiny_overfit_info = _apply_tiny_overfit_if_needed(dataset_train, train_config, logger)

    train_dl_kwargs = get_dataloader_kwargs(train_config, dataset_train, tokenizer, "train")

    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(
        dataset_train,
        num_workers=train_config.num_workers_dataloader,
        pin_memory=True,
        **train_dl_kwargs,
    )

    eval_dataloader = None
    if train_config.run_validation:
        if train_config.batching_strategy == "packing":
            dataset_val = ConcatDataset(dataset_val, chunk_size=train_config.context_length)

        val_dl_kwargs = get_dataloader_kwargs(train_config, dataset_val, tokenizer, "val")

        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            **val_dl_kwargs,
        )

    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
            weight_decay=train_config.weight_decay,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
    # scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lambda step: (
            min(step / train_config.warmup_steps, 1) if step < train_config.warmup_steps
            else  max(0.0, 1 - (step - train_config.warmup_steps) / (train_config.total_steps - train_config.warmup_steps))
            # else 1
        )
    )

    # Start the training process
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        log_config,
        fsdp_config if train_config.enable_fsdp else None,
        local_rank if train_config.enable_fsdp or train_config.enable_ddp else None,
        rank if train_config.enable_fsdp or train_config.enable_ddp else None,
        reference_model=reference_model,
        run_context={
            "policy_init_ckpt_path": kwargs.get("ckpt_path", None),
            "reference_model_path": reference_model_path,
            "tiny_overfit_info": tiny_overfit_info,
        },
    )

    if not (train_config.enable_fsdp or train_config.enable_ddp) or rank==0:
        [logger.info(f'Key: {k}, Value: {v}') for k, v in results.items()]
        if debug_train_sanity:
            infer_ckpt_path = train_config.get("debug_inference_ckpt_path", None)
            if infer_ckpt_path in (None, "", "None"):
                infer_ckpt_path = os.environ.get("SANITY_INFER_CKPT_PATH", None)
            best_ckpt_path = results.get("best_ckpt_path", None)
            logger.info(f"[CKPT] recorded_best_ckpt_path={best_ckpt_path}")
            logger.info(f"[CKPT] inference_loaded_ckpt_path={infer_ckpt_path}")
            if infer_ckpt_path and best_ckpt_path and str(Path(infer_ckpt_path).resolve()) != str(Path(best_ckpt_path).resolve()):
                logger.warning(
                    "[CKPT] [DIAGNOSIS] loaded ckpt path != recorded best ckpt path; inference may not use best checkpoint."
                )

    if not (train_config.enable_fsdp or  train_config.enable_ddp) or rank == 0:
        if log_config.use_wandb:
            wandb.finish()

if __name__ == "__main__":
    main_hydra()
