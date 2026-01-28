"""
Fine-tuning the library models for causal language modeling (GPT, LLaMA, Bloom, ...) on a json file or a dataset.
"""

import json
import math
import os
import sys
import random
import shutil
from dataclasses import dataclass, field
from glob import glob
from types import MethodType
from typing import Literal, Optional, Tuple

import torch
import torch.utils.data
import numpy as np
from datasets import load_dataset
from loguru import logger
from packaging.version import parse as parse_version
from peft import LoraConfig, TaskType, get_peft_model, PeftModel, prepare_model_for_kbit_training
from peft.utils import load_peft_weights, set_peft_model_state_dict
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    Seq2SeqTrainingArguments,
    set_seed,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    __version__ as transformers_version,
)
from transformers.trainer_pt_utils import LabelSmoother
from transformers.utils.versions import require_version

TRAINING_ARGS_NAME = "training_args.bin"
OPTIMIZER_NAME = "optimizer.pt"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"
RNG_STATE_NAME = "rng_state.pth"
TRAINER_STATE_NAME = "trainer_state.json"
PREFIX_CHECKPOINT_DIR = "checkpoint"
from transformers.utils.versions import require_version

from transformers.integrations import is_deepspeed_zero3_enabled

TRANSFORMERS_VERSION = parse_version(transformers_version)

is_flash_attn_2_available = False
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input, unpad_input

    is_flash_attn_2_available = True
except ImportError:
    is_flash_attn_2_available = False

from template import get_conv_template


@dataclass
class ModelArguments:
    """
    模型训练核心配置参数类
    定义了微调或从头训练模型所需的所有关键设置
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "预训练模型权重路径或Hugging Face Hub模型名称，如果需要从头训练模型，请不要设置此参数。"
            )
        },
    )
    load_in_8bit: bool = field(default=False, metadata={"help": "是否以8位量化模式加载模型，可大幅降低显存占用。"})
    load_in_4bit: bool = field(default=False, metadata={"help": "是否以4位量化模式加载模型，显存占用进一步降低。"})
    tokenizer_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "分词器路径或Hugging Face Hub分词器名称。"
                "如果需要从头训练模型，请不要设置此参数。"
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Hugging Face模型缓存目录，用于存储下载的预训练模型文件。"},
    )
    model_revision: Optional[str] = field(
        default="main",
        metadata={"help": "模型版本，可以是分支名、标签名或提交ID。"},
    )
    hf_hub_token: Optional[str] = field(default=None, metadata={"help": "Hugging Face Hub认证令牌，用于访问私有模型。"})
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "是否使用基于tokenizers库的快速分词器，比原生Tokenizer速度更快。"},
    )
    torch_dtype: Optional[str] = field(
        default="float16",
        metadata={
            "help": (
                "模型加载时使用的数据类型。如果设置为'auto'，将自动从模型权重推断数据类型。"
                "可选值：auto, bfloat16, float16, float32"
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    device_map: Optional[str] = field(
        default="auto",
        metadata={"help": "模型设备映射策略。设置为'auto'时将自动选择最佳设备分配方案。"},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "是否信任远程模型中的自定义代码，加载非官方模型时需要设置为True。"},
    )
    rope_scaling: Optional[Literal["linear", "dynamic"]] = field(
        default=None,
        metadata={"help": "采用旋转位置编码缩放策略，支持更长上下文窗口。可选linear或dynamic。"}
    )
    flash_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "启用FlashAttention-2优化，大幅提升训练速度并降低显存占用。"}
    )
    shift_attn: Optional[bool] = field(
        default=False,
        metadata={"help": "启用LongLoRA提出的移位稀疏注意力(S^2-Attn)，进一步优化长文本处理。"}
    )
    neft_alpha: Optional[float] = field(
        default=0,
        metadata={"help": "NEFTune噪声幅度控制参数，建议值为5，用于提升模型泛化能力。"}
    )

    def __post_init__(self):
        if self.model_name_or_path is None:
            raise ValueError("You must specify a valid model_name_or_path to run training.")


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file_dir: Optional[str] = field(default=None, metadata={"help": "The train jsonl data file folder."})
    validation_file_dir: Optional[str] = field(default=None, metadata={"help": "The evaluation jsonl file folder."})
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "If only pad tokens should be ignored. This assumes that `config.pad_token_id` is defined."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=1,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=8,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )

    def __post_init__(self):
        if self.max_train_samples is not None and 0 < self.max_train_samples <= 1000:
            logger.warning("You may set max_train_samples = -1 to run all samples in production.")


@dataclass
class ScriptArguments:
    use_peft: bool = field(default=True, metadata={"help": "Whether to use peft"})
    train_on_inputs: bool = field(default=False, metadata={"help": "Whether to train on inputs"})
    target_modules: Optional[str] = field(default="all")
    lora_rank: Optional[int] = field(default=8)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_alpha: Optional[float] = field(default=32.0)
    modules_to_save: Optional[str] = field(default=None)
    peft_path: Optional[str] = field(default=None, metadata={"help": "The path to the peft model"})
    qlora: bool = field(default=False, metadata={"help": "Whether to use qlora"})
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum model context length. suggest: 8192 * 4, 8192 * 2, 8192, 4096, 2048, 1024, 512"}
    )
    template_name: Optional[str] = field(default="qwen", metadata={"help": "The prompt template name."})
    use_swanlab: bool = field(default=False, metadata={"help": "Whether to use SwanLab for experiment tracking."})

    def __post_init__(self):
        if self.model_max_length < 60:
            raise ValueError("You must specify a valid model_max_length >= 60 to run training")


class SavePeftModelTrainer(Trainer):
    """
    Trainer for lora models with custom checkpoint saving/loading.
    """

    def _atomic_save(self, obj, path):
        tmp_path = f"{path}.tmp"
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)

    def _atomic_save_json_state(self, path):
        tmp_path = f"{path}.tmp"
        self.state.save_to_json(tmp_path)
        os.replace(tmp_path, path)

    def _save_rng_state(self, output_dir):
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state_all()
        self._atomic_save(rng_state, os.path.join(output_dir, RNG_STATE_NAME))

    def save_model(self, output_dir=None, _internal_call=False):
        """Save the LoRA model."""
        if output_dir is None:
            output_dir = self.args.output_dir
            
        # Ensure output_dir exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Save training args
        self._atomic_save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))
        
        # Save adapter to a subdirectory "adapter_model" to keep it clean
        adapter_dir = os.path.join(output_dir, "adapter_model")
        adapter_tmp_dir = f"{adapter_dir}.tmp"
        if os.path.exists(adapter_tmp_dir):
            shutil.rmtree(adapter_tmp_dir)
        self.model.save_pretrained(adapter_tmp_dir)
        if os.path.exists(adapter_dir):
            shutil.rmtree(adapter_dir)
        os.replace(adapter_tmp_dir, adapter_dir)

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        """Load the model from checkpoint."""
        # Load optimizer, scheduler, rng state
        super()._load_from_checkpoint(resume_from_checkpoint, model)
        
        # Load adapter weights
        # Note: model is self.model if not provided
        model = model or self.model
        
        adapter_dir = os.path.join(resume_from_checkpoint, "adapter_model")
        if os.path.exists(adapter_dir):
            logger.info(f"Loading adapter weights from {adapter_dir}")
            # Ensure we are working with a PeftModel
            if isinstance(model, PeftModel):
                # Load the adapter weights into the existing PeftModel
                adapters_weights = load_peft_weights(adapter_dir)
                set_peft_model_state_dict(model, adapters_weights)
            else:
                logger.warning("Model is not a PeftModel, skipping adapter loading.")
        else:
            logger.warning(f"Adapter directory {adapter_dir} not found, skipping adapter loading.")

        optimizer_path = os.path.join(resume_from_checkpoint, OPTIMIZER_NAME)
        scheduler_path = os.path.join(resume_from_checkpoint, SCHEDULER_NAME)
        scaler_path = os.path.join(resume_from_checkpoint, SCALER_NAME)
        rng_path = os.path.join(resume_from_checkpoint, RNG_STATE_NAME)

        if os.path.exists(optimizer_path) and self.optimizer is not None:
            optimizer_state = torch.load(optimizer_path, map_location="cpu")
            self.optimizer.load_state_dict(optimizer_state)
            logger.info(f"Loaded optimizer state from {optimizer_path}")
        else:
            logger.warning(f"Optimizer state not found at {optimizer_path}")

        if os.path.exists(scheduler_path) and self.lr_scheduler is not None:
            scheduler_state = torch.load(scheduler_path, map_location="cpu")
            self.lr_scheduler.load_state_dict(scheduler_state)
            logger.info(f"Loaded scheduler state from {scheduler_path}")
        else:
            logger.warning(f"Scheduler state not found at {scheduler_path}")

        if os.path.exists(scaler_path) and getattr(self, "scaler", None) is not None:
            scaler_state = torch.load(scaler_path, map_location="cpu")
            self.scaler.load_state_dict(scaler_state)
            logger.info(f"Loaded scaler state from {scaler_path}")
        else:
            logger.warning(f"Scaler state not found at {scaler_path}")

        if os.path.exists(rng_path):
            rng_state = torch.load(rng_path, map_location="cpu")
            if "python" in rng_state:
                random.setstate(rng_state["python"])
            if "numpy" in rng_state:
                np.random.set_state(rng_state["numpy"])
            if "torch" in rng_state:
                torch.set_rng_state(rng_state["torch"])
            if "cuda" in rng_state and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(rng_state["cuda"])
            logger.info(f"Loaded rng state from {rng_path}")
        else:
            logger.warning(f"RNG state not found at {rng_path}")

    def _save_checkpoint(self, model, trial, metrics=None):
        if not self.is_world_process_zero():
            return

        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        output_dir = os.path.join(self.args.output_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)

        self.save_model(output_dir)

        if self.optimizer is not None:
            self._atomic_save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
        if self.lr_scheduler is not None:
            self._atomic_save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
        if getattr(self, "scaler", None) is not None:
            self._atomic_save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))

        self._save_rng_state(output_dir)
        swanlab_run_id = os.environ.get("SWANLAB_RUN_ID")
        if swanlab_run_id:
            self.state.swanlab_run_id = swanlab_run_id
        self._atomic_save_json_state(os.path.join(output_dir, TRAINER_STATE_NAME))
        self._atomic_save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        expected_files = [
            os.path.join(output_dir, OPTIMIZER_NAME),
            os.path.join(output_dir, SCHEDULER_NAME),
            os.path.join(output_dir, TRAINER_STATE_NAME),
            os.path.join(output_dir, TRAINING_ARGS_NAME),
            os.path.join(output_dir, RNG_STATE_NAME),
            os.path.join(output_dir, "adapter_model"),
        ]
        missing = [path for path in expected_files if not os.path.exists(path)]
        if missing:
            logger.warning(f"Checkpoint incomplete, missing: {missing}")
        else:
            logger.info(f"Checkpoint saved to {output_dir}")

        if self.args.save_total_limit is not None and self.args.save_total_limit > 0:
            if self.args.output_dir is None:
                logger.warning("output_dir is None, skip rotating checkpoints.")
                return
            self._rotate_checkpoints(use_mtime=True, output_dir=self.args.output_dir)


class SwanLabLoggingCallback(TrainerCallback):
    def __init__(self, swanlab_run=None, is_main_process=True):
        self.swanlab_run = swanlab_run
        self.is_main_process = is_main_process

    def _log(self, metrics):
        if not self.is_main_process or self.swanlab_run is None:
            return
        try:
            if hasattr(self.swanlab_run, "log"):
                self.swanlab_run.log(metrics)
            else:
                import swanlab
                swanlab.log(metrics)
        except Exception as e:
            logger.error(f"❌ SwanLab log failed: {e}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        metrics = {}
        for key in ["loss", "learning_rate", "logits_loss", "logit_loss"]:
            if key in logs:
                metrics[key] = logs[key]
        if metrics:
            metrics["step"] = state.global_step
            self._log(metrics)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        payload = {}
        for key in [
            "eval_loss",
            "eval_accuracy",
            "eval_runtime",
            "eval_samples_per_second",
            "eval_steps_per_second",
            "perplexity",
            "eval_perplexity",
        ]:
            if key in metrics:
                payload[key] = metrics[key]
        if payload:
            payload["step"] = state.global_step
            self._log(payload)


def save_model(model, tokenizer, args):
    """Save the model and the tokenizer."""
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


def save_model_zero3(model, tokenizer, args, trainer):
    """Save the model for deepspeed zero3.
    refer https://github.com/lm-sys/FastChat/blob/main/fastchat/train/train_lora.py#L209
    """
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    state_dict_zero3 = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(args.output_dir, state_dict=state_dict_zero3)
    tokenizer.save_pretrained(output_dir)


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def find_all_linear_names(peft_model, int4=False, int8=False):
    """Find all linear layer names in the model. reference from qlora paper."""
    cls = torch.nn.Linear
    if int4 or int8:
        import bitsandbytes as bnb
        if int4:
            cls = bnb.nn.Linear4bit
        elif int8:
            cls = bnb.nn.Linear8bitLt
    lora_module_names = set()
    for name, module in peft_model.named_modules():
        if isinstance(module, cls):
            # last layer is not add to lora_module_names
            if 'lm_head' in name:
                continue
            if 'output_layer' in name:
                continue
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    return sorted(lora_module_names)


def check_and_optimize_memory():
    """检查并优化GPU内存使用"""
    if not torch.cuda.is_available():
        return

    logger.info("🔍 检查GPU内存状态...")

    # 清理缓存
    torch.cuda.empty_cache()

    # 检查每个GPU的内存状态
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        props = torch.cuda.get_device_properties(i)
        total_memory = props.total_memory / 1024 ** 3
        allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
        cached = torch.cuda.memory_reserved(i) / 1024 ** 3
        free = total_memory - allocated - cached

        logger.info(f"GPU {i} ({props.name}):")
        logger.info(f"  总内存: {total_memory:.1f}GB")
        logger.info(f"  已分配: {allocated:.1f}GB")
        logger.info(f"  已缓存: {cached:.1f}GB")
        logger.info(f"  可用: {free:.1f}GB")

    # 设置内存优化选项
    if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
        torch.backends.cuda.enable_flash_sdp(True)
        logger.info("✅ 启用Flash Attention优化")

    # 启用内存高效的注意力机制
    if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        logger.info("✅ 启用内存高效注意力机制")


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, Seq2SeqTrainingArguments, ScriptArguments))

    # 使用 parse_args_into_dataclasses 时忽略未知参数
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # 如果我们传递了一个 JSON 文件，让我们用它来配置参数
        model_args, data_args, training_args, script_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1]))
    else:
        # 否则解析命令行参数，忽略未知参数
        model_args, data_args, training_args, script_args = parser.parse_args_into_dataclasses(look_for_args_file=False)

    # 确保 DeepSpeed 配置正确加载
    if training_args.deepspeed is not None:
        training_args.distributed_state.deepspeed_plugin = None

    # The Trainer will handle distributed training setup
    is_main_process = training_args.local_rank in [-1, 0]

    # Only log on main process
    if is_main_process:
        logger.info(f"Model args: {model_args}")
        logger.info(f"Data args: {data_args}")
        logger.info(f"Training args: {training_args}")
        logger.info(f"Script args: {script_args}")
        logger.info(
            f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
            + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
        )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    swanlab_run = None
    if script_args.use_swanlab:
        try:
            from swanlab.integration.huggingface import SwanLabCallback
            import swanlab

            is_main_process = training_args.local_rank in [-1, 0]
            if is_main_process:
                api_key = os.environ.get("SWANLAB_API_KEY")
                if not api_key:
                    raise ValueError("SWANLAB_API_KEY is required when --use_swanlab True")
                swanlab.login(api_key=api_key)
                logger.info("🔑 SwanLab logged in via SWANLAB_API_KEY")

                swanlab_run_id = os.environ.get("SWANLAB_RUN_ID")
                if not swanlab_run_id and training_args.resume_from_checkpoint:
                    trainer_state_path = os.path.join(training_args.resume_from_checkpoint, TRAINER_STATE_NAME)
                    if os.path.exists(trainer_state_path):
                        with open(trainer_state_path, "r", encoding="utf-8") as f:
                            trainer_state = json.load(f)
                        swanlab_run_id = trainer_state.get("swanlab_run_id")

                run_name = f"{model_args.model_name_or_path.split('/')[-1]}-{script_args.template_name}"
                resume = "must" if swanlab_run_id else None
                run = swanlab.init(project="MedicalGPT-SFT", name=run_name, id=swanlab_run_id, resume=resume)
                swanlab_run = run
                run_id = getattr(run, "id", None) if run is not None else None
                if run_id:
                    os.environ["SWANLAB_RUN_ID"] = str(run_id)
                logger.info("🦢 Enabling SwanLab integration on Rank 0...")
                swanlab_callback = SwanLabCallback(
                    project="MedicalGPT-SFT",
                    experiment_name=run_name,
                    config={
                        "model_args": model_args.__dict__,
                        "data_args": data_args.__dict__,
                        "training_args": training_args.__dict__,
                        "script_args": script_args.__dict__,
                    }
                )
            else:
                swanlab_callback = None
        except ImportError:
            if training_args.local_rank in [-1, 0]:
                logger.error("❌ SwanLab is not installed. Please install it via `pip install swanlab`.")
            swanlab_callback = None
        except Exception as e:
            logger.error(f"❌ SwanLab initialization failed: {e}")
            swanlab_callback = None
    else:
        swanlab_callback = None

    # Load tokenizer
    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "trust_remote_code": model_args.trust_remote_code,
    }
    tokenizer_name_or_path = model_args.tokenizer_name_or_path
    if not tokenizer_name_or_path:
        tokenizer_name_or_path = model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, **tokenizer_kwargs)
    prompt_template = get_conv_template(script_args.template_name)
    if tokenizer.eos_token_id is None:
        tokenizer.eos_token = prompt_template.stop_str  # eos token is required
        tokenizer.add_special_tokens({"eos_token": tokenizer.eos_token})
        logger.info(f"Add eos_token: {tokenizer.eos_token}, eos_token_id: {tokenizer.eos_token_id}")
    if tokenizer.bos_token_id is None:
        tokenizer.add_special_tokens({"bos_token": tokenizer.eos_token})
        tokenizer.bos_token_id = tokenizer.eos_token_id
        logger.info(f"Add bos_token: {tokenizer.bos_token}, bos_token_id: {tokenizer.bos_token_id}")
    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token_id is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.pad_token = tokenizer.eos_token
        logger.info(f"Add pad_token: {tokenizer.pad_token}, pad_token_id: {tokenizer.pad_token_id}")
    logger.debug(f"Tokenizer: {tokenizer}")

    IGNORE_INDEX = LabelSmoother.ignore_index if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id

    # Get datasets
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
        )
        if "validation" not in raw_datasets.keys():
            shuffled_train_dataset = raw_datasets["train"].shuffle(seed=42)
            # Split the shuffled train dataset into training and validation sets
            split = shuffled_train_dataset.train_test_split(
                test_size=data_args.validation_split_percentage / 100,
                seed=42
            )
            # Assign the split datasets back to raw_datasets
            raw_datasets["train"] = split["train"]
            raw_datasets["validation"] = split["test"]
    else:
        # Loading a dataset from local files.
        data_files = {}
        if data_args.train_file_dir is not None and os.path.exists(data_args.train_file_dir):
            if os.path.isdir(data_args.train_file_dir):
                train_data_files = glob(f'{data_args.train_file_dir}/**/*.json', recursive=True) + glob(
                    f'{data_args.train_file_dir}/**/*.jsonl', recursive=True)
            else:
                train_data_files = [data_args.train_file_dir]
            logger.info(f"train files: {train_data_files}")
            data_files["train"] = train_data_files
        if data_args.validation_file_dir is not None and os.path.exists(data_args.validation_file_dir):
            if os.path.isdir(data_args.validation_file_dir):
                eval_data_files = glob(f'{data_args.validation_file_dir}/**/*.json', recursive=True) + glob(
                    f'{data_args.validation_file_dir}/**/*.jsonl', recursive=True)
            else:
                eval_data_files = [data_args.validation_file_dir]
            logger.info(f"eval files: {eval_data_files}")
            data_files["validation"] = eval_data_files
        raw_datasets = load_dataset(
            'json',
            data_files=data_files,
            cache_dir=model_args.cache_dir,
        )
        # If no validation data is there, validation_split_percentage will be used to divide the dataset.
        if "validation" not in raw_datasets.keys():
            shuffled_train_dataset = raw_datasets["train"].shuffle(seed=42)
            split = shuffled_train_dataset.train_test_split(
                test_size=float(data_args.validation_split_percentage / 100),
                seed=42
            )
            raw_datasets["train"] = split["train"]
            raw_datasets["validation"] = split["test"]
    logger.info(f"Raw datasets: {raw_datasets}")

    # Preprocessing the datasets
    max_length = script_args.model_max_length

    def preprocess_function(examples):
        """
        Preprocessing the datasets.
            part of code modified from https://github.com/lm-sys/FastChat
        """
        input_ids_list = []
        attention_mask_list = []
        targets_list = []
        roles = ["human", "gpt"]

        def get_dialog(examples):
            system_prompts = examples.get("system_prompt", "")
            if 'conversations' in examples:
                for i, source in enumerate(examples['conversations']):
                    system_prompt = ""
                    if len(source) < 2:
                        continue
                    data_role = source[0].get("from", "")
                    if data_role == "system":
                        # Skip the first one if it is from system
                        system_prompt = source[0]["value"]
                        source = source[1:]
                        data_role = source[0].get("from", "")
                    if data_role not in roles or data_role != roles[0]:
                        # Skip the first one if it is not from human
                        source = source[1:]
                    if len(source) < 2:
                        continue
                    messages = []
                    for j, sentence in enumerate(source):
                        data_role = sentence.get("from", "")
                        if data_role not in roles:
                            logger.warning(f"unknown role: {data_role}, {i}. (ignored)")
                            break
                        if data_role == roles[j % 2]:
                            messages.append(sentence["value"])
                    if len(messages) % 2 != 0:
                        continue
                    # Convert the list to pairs of elements
                    history_messages = [[messages[k], messages[k + 1]] for k in range(0, len(messages), 2)]
                    if not system_prompt:
                        system_prompt = system_prompts[i] if system_prompts else ""
                    yield prompt_template.get_dialog(history_messages, system_prompt=system_prompt)
            elif 'instruction' in examples and 'output' in examples:
                for i, (instruction, inp, output) in enumerate(
                        zip(examples['instruction'], examples.get('input', [""] * len(examples['instruction'])),
                            examples['output'])):
                    if inp and len(str(inp).strip()) > 0:
                        instruction = instruction + '\n\n' + str(inp)
                    history_messages = [[instruction, output]]
                    system_prompt = system_prompts[i] if system_prompts else ""
                    yield prompt_template.get_dialog(history_messages, system_prompt=system_prompt)
            else:
                logger.warning(f"Unknown data format: {examples.keys()}")

        for dialog in get_dialog(examples):
            input_ids, labels = [], []

            for i in range(len(dialog) // 2):
                source_ids = tokenizer.encode(text=dialog[2 * i], add_special_tokens=(i == 0))
                target_ids = tokenizer.encode(text=dialog[2 * i + 1], add_special_tokens=False)

                total_len = len(source_ids) + len(target_ids)
                max_source_len = int(max_length * (len(source_ids) / total_len))
                max_target_len = int(max_length * (len(target_ids) / total_len))

                if len(source_ids) > max_source_len:
                    source_ids = source_ids[:max_source_len]
                if len(target_ids) > max_target_len - 1:  # eos token
                    target_ids = target_ids[:max_target_len - 1]
                if len(source_ids) > 0 and source_ids[0] == tokenizer.eos_token_id:
                    source_ids = source_ids[1:]
                if len(target_ids) > 0 and target_ids[-1] == tokenizer.eos_token_id:
                    target_ids = target_ids[:-1]
                if len(input_ids) + len(source_ids) + len(target_ids) + 1 > max_length:
                    break

                input_ids += source_ids + target_ids + [tokenizer.eos_token_id]  # add eos token for each turn
                if script_args.train_on_inputs:
                    labels += source_ids + target_ids + [tokenizer.eos_token_id]
                else:
                    labels += [IGNORE_INDEX] * len(source_ids) + target_ids + [tokenizer.eos_token_id]

            input_ids_list.append(input_ids)
            attention_mask_list.append([1] * len(input_ids))
            targets_list.append(labels)

        return dict(
            input_ids=input_ids_list,
            attention_mask=attention_mask_list,
            labels=targets_list,
        )

    def filter_empty_labels(example):
        """Remove empty labels dataset."""
        return not all(label == IGNORE_INDEX for label in example["labels"])

    train_dataset = None
    max_train_samples = 0
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets['train'].shuffle(seed=42)
        max_train_samples = len(train_dataset)
        if data_args.max_train_samples is not None and data_args.max_train_samples > 0:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

        if is_main_process:
            logger.debug(f"Example train_dataset[0]: {train_dataset[0]}")

        with training_args.main_process_first(desc="Train dataset tokenization"):
            tokenized_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=train_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset" if is_main_process else None,
            )
            train_dataset = tokenized_dataset.filter(
                filter_empty_labels,
                num_proc=data_args.preprocessing_num_workers
            )

            if is_main_process:
                logger.debug(f"Num train_samples: {len(train_dataset)}")
                logger.debug("Tokenized training example:")
                logger.debug(f"Decode input_ids[0]:\n{tokenizer.decode(train_dataset[0]['input_ids'])}")
                replaced_labels = [label if label != IGNORE_INDEX else tokenizer.pad_token_id
                                   for label in list(train_dataset[0]['labels'])]
                logger.debug(f"Decode labels[0]:\n{tokenizer.decode(replaced_labels)}")

    eval_dataset = None
    max_eval_samples = 0
    if training_args.do_eval:
        with training_args.main_process_first(desc="Eval dataset tokenization"):
            if "validation" not in raw_datasets:
                raise ValueError("--do_eval requires a validation dataset")
            eval_dataset = raw_datasets["validation"]
            max_eval_samples = len(eval_dataset)
            if data_args.max_eval_samples is not None and data_args.max_eval_samples > 0:
                max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                eval_dataset = eval_dataset.select(range(max_eval_samples))
            eval_size = len(eval_dataset)
            logger.debug(f"Num eval_samples: {eval_size}")
            if eval_size > 500:
                logger.warning(f"Num eval_samples is large: {eval_size}, "
                               f"training slow, consider reduce it by `--max_eval_samples=50`")
            logger.debug(f"Example eval_dataset[0]: {eval_dataset[0]}")
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=eval_dataset.column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )
            eval_dataset = eval_dataset.filter(filter_empty_labels, num_proc=data_args.preprocessing_num_workers)
            logger.debug(f"Num eval_samples: {len(eval_dataset)}")
            logger.debug("Tokenized eval example:")
            logger.debug(tokenizer.decode(eval_dataset[0]['input_ids']))

    # Load model
    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        ddp = world_size > 1
        if is_deepspeed_zero3_enabled() or ddp:
            device_map = None
        else:
            device_map = model_args.device_map
        if script_args.qlora and (len(training_args.fsdp) > 0 or is_deepspeed_zero3_enabled()):
            logger.warning("FSDP and DeepSpeed ZeRO-3 are both currently incompatible with QLoRA.")

        config_kwargs = {
            "trust_remote_code": model_args.trust_remote_code,
            "cache_dir": model_args.cache_dir,
            "revision": model_args.model_revision,
            "token": model_args.hf_hub_token,
        }
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)

        # Set RoPE scaling
        if model_args.rope_scaling is not None:
            if hasattr(config, "rope_scaling"):
                if model_args.rope_scaling == "dynamic":
                    logger.warning(
                        "Dynamic NTK may not work well with fine-tuning. "
                        "See: https://github.com/huggingface/transformers/pull/24653"
                    )
                current_max_length = getattr(config, "max_position_embeddings", None)
                if current_max_length and script_args.model_max_length > current_max_length:
                    scaling_factor = float(math.ceil(script_args.model_max_length / current_max_length))
                else:
                    logger.warning(f"The model_max_length({script_args.model_max_length}) is smaller than max "
                                   f"length({current_max_length}). Consider increase model_max_length.")
                    scaling_factor = 1.0

                setattr(config, "rope_scaling", {"type": model_args.rope_scaling, "factor": scaling_factor})
                logger.info("Using {} scaling strategy and setting scaling factor to {}".format(
                    model_args.rope_scaling, scaling_factor
                ))
            else:
                logger.warning("Current model does not support RoPE scaling.")

        # Set FlashAttention-2
        if model_args.flash_attn:
            if is_flash_attn_2_available:
                config_kwargs["use_flash_attention_2"] = True
                logger.info("Using FlashAttention-2 for faster training and inference.")
            else:
                logger.warning("FlashAttention-2 is not installed.")
        elif model_args.shift_attn and getattr(config, "model_type", None) == "llama":
            logger.warning("Using `--flash_attn` for faster training in large context length, enable if your GPU"
                           " is RTX3090, RTX4090, A100 or H100.")

        # Set shifted sparse attention (S^2-Attn)
        if model_args.shift_attn:
            if getattr(config, "model_type", None) == "llama":
                setattr(config, "group_size_ratio", 0.25)
                logger.info("Using shifted sparse attention with group_size_ratio=1/4.")
            else:
                logger.warning("Current model does not support shifted sparse attention.")

        load_in_4bit = model_args.load_in_4bit
        load_in_8bit = model_args.load_in_8bit
        quantization_config = None
        if load_in_4bit and load_in_8bit:
            raise ValueError("Error, load_in_4bit and load_in_8bit cannot be set at the same time")
        elif load_in_8bit or load_in_4bit:
            logger.info(f"Quantizing model, load_in_4bit: {load_in_4bit}, load_in_8bit: {load_in_8bit}")
            if is_deepspeed_zero3_enabled():
                raise ValueError("DeepSpeed ZeRO-3 is incompatible with quantization.")
            if load_in_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            elif load_in_4bit:
                if script_args.qlora:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch_dtype,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4"
                    )
                else:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch_dtype,
                    )

        model_kwargs = {
            "config": config,
            "torch_dtype": torch_dtype,
            "trust_remote_code": model_args.trust_remote_code,
            "quantization_config": quantization_config,
            "low_cpu_mem_usage": True,  # 减少CPU内存使用
        }
        if device_map is not None:
            model_kwargs["device_map"] = device_map

        # 设置device_map
        if device_map == 'auto':
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                # 大模型多GPU：使用auto进行张量并行
                model_kwargs["device_map"] = "auto"
                # 设置最大内存使用
                max_memory = {}
                for i in range(num_gpus):
                    # 为每个GPU预留一些内存给梯度和优化器
                    gpu_props = torch.cuda.get_device_properties(i)
                    total_mem = gpu_props.total_memory
                    # 预留20%内存给训练时的梯度、优化器状态等
                    usable_mem = int(total_mem * 0.8)
                    max_memory[i] = f"{usable_mem // (1024 ** 3)}GiB"

                model_kwargs["max_memory"] = max_memory

        logger.info(f"🔧 大模型训练配置:")
        logger.info(f"  model_kwargs: {model_kwargs}")

        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs
        )

        logger.info("✅ 模型加载完成")

        # 显示模型分布信息
        logger.info("📊 模型分布情况:")
        if hasattr(model, 'hf_device_map') and model.hf_device_map:
            logger.info("🔧 使用HuggingFace设备映射:")
            for module_name, device in model.hf_device_map.items():
                logger.info(f"  {module_name}: {device}")

            # 统计每个GPU上的模块数量
            device_count = {}
            for device in model.hf_device_map.values():
                device_str = str(device)
                device_count[device_str] = device_count.get(device_str, 0) + 1

            logger.info("📈 设备使用统计:")
            for device, count in device_count.items():
                logger.info(f"  {device}: {count} 个模块")
        else:
            # 检查模型参数的设备分布
            device_params = {}
            total_params = 0
            for name, param in model.named_parameters():
                device = str(param.device)
                if device not in device_params:
                    device_params[device] = {'count': 0, 'size': 0}
                device_params[device]['count'] += 1
                device_params[device]['size'] += param.numel()
                total_params += param.numel()

            logger.info("📈 参数设备分布:")
            for device, info in device_params.items():
                param_size_gb = info['size'] * 4 / 1024 ** 3  # 假设float32
                percentage = info['size'] / total_params * 100
                logger.info(f"  {device}: {info['count']} 个参数组, {param_size_gb:.2f}GB ({percentage:.1f}%)")

        # 显示GPU内存使用情况
        if torch.cuda.is_available():
            logger.info("💾 GPU内存使用情况:")
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
                cached = torch.cuda.memory_reserved(i) / 1024 ** 3
                total = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3
                logger.info(f"  GPU {i}: 已分配={allocated:.1f}GB, 缓存={cached:.1f}GB, 总计={total:.1f}GB")

        # Fix ChatGLM2 and ChatGLM3 and internlm2 LM head
        if getattr(config, "model_type", None) == "chatglm" or getattr(config, "model_type", None) == "internlm2":
            setattr(model, "lm_head", model.transformer.output_layer)
            setattr(model, "_keys_to_ignore_on_save", ["lm_head.weight"])

        # Set NEFTune trick for fine-tuning
        if model_args.neft_alpha > 0:
            input_embed = model.get_input_embeddings()
            if isinstance(input_embed, torch.nn.Embedding):
                def noisy_forward(self: torch.nn.Embedding, x: torch.Tensor) -> torch.Tensor:
                    embeddings = input_embed.__class__.forward(self, x)
                    dims = self.num_embeddings * self.embedding_dim
                    mag_norm = model_args.neft_alpha / (dims ** 0.5)
                    embeddings += torch.zeros_like(embeddings).uniform_(-mag_norm, mag_norm)
                    return embeddings

                input_embed.forward = MethodType(noisy_forward, input_embed)
                logger.info("Using noisy embedding with alpha={:.2f}".format(model_args.neft_alpha))
            else:
                logger.warning("Input embeddings are not normal nn.Embedding, cannot transform into noisy embedding.")

        # Patch Mixtral MOE model
        if getattr(config, "model_type", None) == "mixtral" and is_deepspeed_zero3_enabled():
            require_version("deepspeed>=0.13.0", "To fix: pip install deepspeed>=0.13.0")
            if TRANSFORMERS_VERSION < parse_version("4.50.0"):
                from deepspeed.utils import set_z3_leaf_modules  # type: ignore
                from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock  # type: ignore

                set_z3_leaf_modules(model, [MixtralSparseMoeBlock])
            else:
                logger.info(f"Transformers {transformers_version} detected, skip Mixtral ZeRO-3 patch.")

        # Patch DeepSeek-V3 MoE module
        if getattr(config, "model_type", None) == "deepseek_v3" and is_deepspeed_zero3_enabled():
            require_version("deepspeed>=0.13.0", "To fix: pip install deepspeed>=0.13.0")
            if TRANSFORMERS_VERSION < parse_version("4.50.0"):
                for layer in model.model.layers:
                    if 'DeepseekV3MoE' in str(type(layer.mlp)):
                        layer.mlp._z3_leaf = True
            else:
                logger.info(f"Transformers {transformers_version} detected, skip DeepSeek-V3 ZeRO-3 patch.")
    else:
        raise ValueError(f"Error, model_name_or_path is None, SFT must be loaded from a pre-trained model")

    if script_args.use_peft:
        logger.info("Fine-tuning method: LoRA(PEFT)")

        # Set fp32 forward hook for lm_head
        output_layer = getattr(model, "lm_head")
        if isinstance(output_layer, torch.nn.Linear) and output_layer.weight.dtype != torch.float32:
            def fp32_forward_post_hook(module: torch.nn.Module, args: Tuple[torch.Tensor], output: torch.Tensor):
                return output.to(torch.float32)

            output_layer.register_forward_hook(fp32_forward_post_hook)

        # Load LoRA model
        if script_args.peft_path is not None:
            logger.info(f"Peft from pre-trained model: {script_args.peft_path}")
            model = PeftModel.from_pretrained(model, script_args.peft_path, is_trainable=True)
        else:
            logger.info("Init new peft model")
            if load_in_8bit or load_in_4bit:
                model = prepare_model_for_kbit_training(model, training_args.gradient_checkpointing)
            target_modules = script_args.target_modules.split(',') if script_args.target_modules else None
            if target_modules and 'all' in target_modules:
                target_modules = find_all_linear_names(model, int4=load_in_4bit, int8=load_in_8bit)
            modules_to_save = script_args.modules_to_save
            if modules_to_save is not None:
                modules_to_save = modules_to_save.split(',')
            logger.info(f"Peft target_modules: {target_modules}")
            logger.info(f"Peft lora_rank: {script_args.lora_rank}")
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=script_args.lora_rank,
                lora_alpha=script_args.lora_alpha,
                lora_dropout=script_args.lora_dropout,
                modules_to_save=modules_to_save)
            model = get_peft_model(model, peft_config)
        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.float32)
        model.print_trainable_parameters()
    else:
        logger.info("Fine-tuning method: Full parameters training")
        model = model.float()
        print_trainable_parameters(model)

    # Initialize our Trainer
    if training_args.gradient_checkpointing and getattr(model, "supports_gradient_checkpointing", False):
        gradient_checkpointing_kwargs = {"use_reentrant": False} if ddp else {}
        try:
            if gradient_checkpointing_kwargs:
                model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            else:
                model.gradient_checkpointing_enable()
        except TypeError:
            model.gradient_checkpointing_enable()
        model.config.use_cache = False
        logger.info("Gradient checkpointing enabled.")
    else:
        model.config.use_cache = True
        logger.info("Gradient checkpointing disabled.")
    model.enable_input_require_grads()
    if not ddp and torch.cuda.device_count() > 1:
        # Keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=IGNORE_INDEX,
        pad_to_multiple_of=4 if tokenizer.padding_side == "right" else None,  # for shifted sparse attention
    )
    callbacks = []
    if swanlab_callback:
        callbacks.append(swanlab_callback)
    if swanlab_run and is_main_process:
        callbacks.append(SwanLabLoggingCallback(swanlab_run=swanlab_run, is_main_process=is_main_process))
    trainer = SavePeftModelTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks if callbacks else None,
    )

    # Training
    if training_args.do_train:
        if trainer.is_world_process_zero():
            logger.info("*** Train ***")
            sample = next(iter(trainer.get_train_dataloader()))
            logger.debug(f"Train dataloader example: {sample}")
            logger.debug(f"input_ids:\n{list(sample['input_ids'])[:3]}, \nlabels:\n{list(sample['labels'])[:3]}")
            logger.debug(f"Decode input_ids[0]:\n{tokenizer.decode(sample['input_ids'][0])}")
            replaced_labels = [label if label != IGNORE_INDEX else tokenizer.pad_token_id for label in
                               sample['labels'][0]]
            logger.debug(f"Decode labels[0]:\n{tokenizer.decode(replaced_labels)}")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        metrics = train_result.metrics
        metrics["train_samples"] = max_train_samples
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        model.config.use_cache = True  # enable cache after training
        tokenizer.padding_side = "left"  # restore padding side
        tokenizer.init_kwargs["padding_side"] = "left"

        # if trainer.is_world_process_zero():
        #     logger.debug(f"Training metrics: {metrics}")
        #     logger.info(f"Saving model checkpoint to {training_args.output_dir}")
        #     if is_deepspeed_zero3_enabled():
        #         save_model_zero3(model, tokenizer, training_args, trainer)
        #     else:
        #         save_model(model, tokenizer, training_args)

    # Evaluation
    if training_args.do_eval:
        if trainer.is_world_process_zero():
            logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")

        metrics["eval_samples"] = max_eval_samples
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        metrics["eval_perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        if trainer.is_world_process_zero():
            logger.debug(f"Eval metrics: {metrics}")


if __name__ == "__main__":
    main()
