import os.path as osp
from argparse import Namespace
from typing import Any, List, Tuple

from transformers import HfArgumentParser  # type: ignore

from lmlib.arguments import (
    DataArguments,
    InferenceArguments,
    LoraArguments,
    ModelArguments,
)
from lmlib.arguments import PreTrainingArguments as TrainingArguments

base_dir = "/usr1/data/ruohongz/ds"
save_dir = f"{base_dir}/save"
cache_dir = f"{base_dir}/cache"
data_dir = f"{base_dir}/data"

exp_dict_default = Namespace(
    data_name="info_eli_dawson.json",
    model_name="decapoda-research/llama-13b-hf",
    lora_model_name=osp.join("eli-lora-llama-13b-v1"),
)


def set_path(exp_dict: Any = exp_dict_default) -> Namespace:
    data_name = exp_dict.data_name

    paths = Namespace(
        base_dir=base_dir,
        save_dir=save_dir,
        cache_dir=cache_dir,
        data_dir=data_dir,
        data_name=data_name,
        model_name=exp_dict.model_name,
        lora_model_name=exp_dict.lora_model_name,
        lora_path=osp.join(save_dir, exp_dict.lora_model_name),
    )
    return paths


def get_args(paths: Namespace) -> List[str]:
    args = [
        "--exp_name",
        "seed100",
        "--seed",
        "100",
        "--data_dir",
        paths.data_dir,
        "--output_dir",
        paths.save_dir,
        "--cache_dir",
        paths.cache_dir,
        "--max_steps",
        "500",
        "--model_name_or_path",
        paths.model_name,
        "--per_device_eval_batch_size",
        "32",
        "--per_device_train_batch_size",
        "32",
        "--num_train_epochs",
        "1",
        "--learning_rate",
        "1e-5",
        "--max_text_length",
        "128",
        "--dataloader_num_workers",
        "1",
        "--model_max_length",
        "2048",
    ]
    return args


def setup_args(
    exp_dict: Namespace = exp_dict_default, additional_args: List[str] = []
) -> Tuple[Namespace, Namespace, Namespace, Namespace]:
    paths = set_path(exp_dict)
    args = get_args(paths)
    parser = HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses(args + additional_args)
    return model_args, data_args, training_args, lora_args


def get_inf_args(paths: Namespace) -> List[str]:
    args = [
        "--model_path",
        paths.model_name,
        "--cache_dir",
        paths.cache_dir,
        "--lora_weight_path",
        paths.lora_path,
        "--load_8bit",
    ]
    return args


def setup_inf_args(
    exp_dict: Namespace = exp_dict_default, additional_args: List[str] = []
) -> Tuple[Namespace, Namespace]:
    paths = set_path(exp_dict)
    args = get_inf_args(paths)
    parser = HfArgumentParser((InferenceArguments, LoraArguments))
    inf_args, lora_args = parser.parse_args_into_dataclasses(args + additional_args)
    return inf_args, lora_args
