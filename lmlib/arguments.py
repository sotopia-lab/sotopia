from dataclasses import dataclass, field
from typing import Optional, Union

from transformers.training_args import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        }
    )
    model_name_short: Union[str, None] = field(default=None)
    gen_model_name: Union[str, None] = field(default=None)
    gen_model_name_short: Union[str, None] = field(default=None)
    target_model_path: Union[str, None] = field(
        default=None,
        metadata={"help": "Path to pretrained reranker target model"},
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )

    # out projection
    add_pooler: bool = field(default=False)
    projection_in_dim: int = field(default=768)
    projection_out_dim: int = field(default=768)

    decay_rate: Optional[float] = field(
        default=0.6, metadata={"help": "Decay learning rate"}
    )
    pooling: Optional[str] = field(
        default="pooler", metadata={"help": "Pooling method"}
    )
    device: Optional[str] = field(default="cuda:1", metadata={"help": "Device to use"})
    gen_device: Optional[str] = field(
        default="cuda:0", metadata={"help": "Device to use"}
    )
    max_text_length: int = field(default=128)


@dataclass
class DataArguments:
    # data dir, path, name related
    train_dir: Union[str, None] = field(
        default=None, metadata={"help": "Path to train directory"}
    )
    train_path: Union[str, None] = field(
        default=None, metadata={"help": "Path to single train file"}
    )
    eval_path: Union[str, None] = field(
        default=None, metadata={"help": "Path to eval file"}
    )
    corpus_path: Union[str, None] = field(
        default=None, metadata={"help": "Path to corpus file"}
    )
    data_dir: Union[str, None] = field(
        default=None, metadata={"help": "Path to data directory"}
    )
    data_path: Union[str, None] = field(
        default=None, metadata={"help": "Path to the single data file"}
    )
    processed_data_path: Union[str, None] = field(
        default=None, metadata={"help": "Path to processed data directory"}
    )
    context_path: Union[str, None] = field(default=None)
    dataset_name: Union[str, None] = field(
        default=None, metadata={"help": "huggingface dataset name"}
    )
    prompt_path: Union[str, None] = field(default=None)
    label_name_path: Union[str, None] = field(default=None)
    label_path: Union[str, None] = field(default=None)
    save_path: Union[str, None] = field(default=None)

    # data format related
    data_cache_dir: Union[str, None] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the data downloaded from huggingface"
        },
    )
    exp_name: str = field(default="exp")
    conv_template: Union[str, None] = field(default=None)


@dataclass
class PreTrainingArguments(TrainingArguments):  # type: ignore[misc]
    # generate
    generation_mode: Union[str, None] = field(default=None)
    load_large_model: bool = field(default=False)
    num_samples: int = field(default=5)

    top_k: int = field(default=10)
    top_p: float = field(default=0.9)
    temperature: float = field(default=1.0)

    gen_batch_size: int = field(default=45)
    max_length: Union[str, None] = field(default=None)
    min_length: Union[str, None] = field(default=None)
    do_sample: bool = field(default=True)
    num_beams: int = field(default=1)

    max_iter: int = field(default=1000)
    model_max_length: int = field(default=512)


@dataclass
class LoraArguments:
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    # lora_target_modules: typing.List[str] = field(
    #     default_factory=lambda: ["q_proj", "v_proj"]
    # )
    lora_target_modules: str = "q_proj,v_proj"
    lora_weight_path: str = ""
    bias: str = "none"


@dataclass
class InferenceArguments:
    model_path: str = field(default="decapoda-research/llama-13b-hf")
    cache_dir: Union[str, None] = field(default=None)
    device: str = field(default="cuda")
    gpus: Union[str, None] = field(default=None)
    num_gpus: str = field(default="1")
    conv_template: Union[str, None] = field(default=None)
    temperature: float = field(default=0.7)
    max_new_tokens: int = field(default=512)
    style: str = field(default="simple")
    load_8bit: bool = field(default=False)
    max_gpu_memory: bool = field(default=False)
    debug: bool = field(default=False)
