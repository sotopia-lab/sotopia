"""Inference for FastChat models."""

import abc
import os
import os.path as osp
from typing import Any, Dict, Optional, Union

import torch
from logzero import logger
from peft import PeftModel, set_peft_model_state_dict
from transformers import LlamaTokenizer  # type: ignore[attr-defined]

from lmlib.utils.conversation import (
    SeparatorStyle,
    compute_skip_echo_len,
    conv_templates,
    get_default_conv_template,
)
from lmlib.utils.data_utils import load_json

# try:
#     from transformers import (
#         AutoModel,
#         AutoModelForCausalLM,
#         AutoModelForSeq2SeqLM,
#         LlamaForCausalLM,
#         LlamaTokenizer,
#     )
# except ImportError:
#     from transformers import (
#         AutoModelForCausalLM,
#         LLaMATokenizer,
#         LLamaForCausalLM,
#         AutoModel,
#         AutoModelForSeq2SeqLM,
#     )


# from lmlib.serve.compression import compress_module
# from lmlib.serve.monkey_patch_non_inplace import (
#     replace_llama_attn_with_non_inplace_operations,
# )
# from lmlib.serve.serve_chatglm import chatglm_generate_stream


def get_gpu_memory(max_gpus: Union[int, None] = None) -> list[float]:
    gpu_memory = []
    num_gpus = (
        torch.cuda.device_count()
        if max_gpus is None
        else min(max_gpus, torch.cuda.device_count())
    )

    for gpu_id in range(num_gpus):
        with torch.cuda.device(gpu_id):
            device = torch.cuda.current_device()
            gpu_properties = torch.cuda.get_device_properties(device)
            total_memory = gpu_properties.total_memory / (1024**3)
            allocated_memory = torch.cuda.memory_allocated() / (1024**3)
            available_memory = total_memory - allocated_memory
            gpu_memory.append(available_memory)
    return gpu_memory


def load_gen_model(
    gen_model_name: str,
    cache_dir: str,
    large_model: bool = False,
    device: str = "cuda",
    lora_path: Union[str, None] = None,
    margs: Dict[str, Any] = {},
) -> Any:
    from lmlib.model_tools import (
        load_and_cache_large_model,
        load_and_cache_model,
    )

    if large_model:
        gen_model = load_and_cache_large_model(
            gen_model_name, cache_dir=cache_dir, device=device
        )
    else:
        # margs = {"revision": "float16", "torch_dtype": torch.float16, "low_cpu_mem_usage": True}
        gen_model = load_and_cache_model(
            gen_model_name, cache_dir=cache_dir, margs=margs
        )
        gen_model = gen_model.to(device)
    if lora_path is not None:
        logger.info(f"load lora from {lora_path}")
        from argparse import Namespace

        from peft import get_peft_model
        from transformers.trainer_utils import get_last_checkpoint

        lora_config_json = load_json(osp.join(lora_path, "adapter_config.json"))
        lora_config = Namespace(**lora_config_json)
        gen_model = get_peft_model(gen_model, lora_config)

        ckpt_path = get_last_checkpoint(lora_path)
        if ckpt_path is None:
            gen_model = PeftModel.from_pretrained(
                gen_model,
                lora_path,
                torch_dtype=torch.float16,
            )
        else:
            checkpoint_name = os.path.join(ckpt_path, "pytorch_model.bin")
            adapter_weigths = torch.load(checkpoint_name)
            set_peft_model_state_dict(gen_model, adapter_weigths)

    gen_tokenizer = LlamaTokenizer.from_pretrained(gen_model_name, cache_dir=cache_dir)
    # use the vicuna version of special tokens
    gen_tokenizer.add_special_tokens(
        {
            "bos_token": "<s>",
            "eos_token": "</s>",
            "unk_token": "<unk>",
            "pad_token": "<unk>",
        }
    )
    return gen_model, gen_tokenizer


@torch.inference_mode()
def generate_stream(
    model: Any,
    tokenizer: Any,
    params: Any,
    device: str,
    context_len: int = 2048,
    stream_interval: int = 2,
) -> Any:
    prompt = params["prompt"]
    l_prompt = len(prompt)
    temperature = float(params.get("temperature", 1.0))
    max_new_tokens = int(params.get("max_new_tokens", 256))
    stop_str = params.get("stop", None)
    stop_token_ids = params.get("stop_ids", [tokenizer.eos_token_id])

    input_ids = tokenizer(prompt).input_ids
    output_ids = list(input_ids)

    max_src_len = context_len - max_new_tokens - 8
    input_ids = input_ids[-max_src_len:]
    token = 0

    for i in range(max_new_tokens):
        if i == 0:
            if model.config.is_encoder_decoder:
                encoder_outputs = model.encoder(
                    input_ids=torch.as_tensor([input_ids], device=device)
                )
                out = model(
                    torch.as_tensor([input_ids], device=device),
                    decoder_input_ids=torch.as_tensor(
                        [[model.generation_config.decoder_start_token_id]],
                        device=device,
                    ),
                    encoder_outputs=encoder_outputs,
                    use_cache=True,
                )
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                out = model(torch.as_tensor([input_ids], device=device), use_cache=True)
                logits = out.logits
                past_key_values = out.past_key_values
        else:
            if model.config.is_encoder_decoder:
                out = model(
                    input_ids=torch.as_tensor([input_ids], device=device),
                    use_cache=True,
                    encoder_outputs=encoder_outputs,
                    decoder_input_ids=torch.as_tensor([[token]], device=device),
                    past_key_values=past_key_values,
                )
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                out = model(
                    input_ids=torch.as_tensor([[token]], device=device),
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                logits = out.logits
                past_key_values = out.past_key_values

        last_token_logits = logits[0][-1]

        if device == "mps":
            # Switch to CPU by avoiding some bugs in mps backend.
            last_token_logits = last_token_logits.float().to("cpu")

        if temperature < 1e-4:
            token = int(torch.argmax(last_token_logits))
        else:
            probs = torch.softmax(last_token_logits / temperature, dim=-1)
            token = int(torch.multinomial(probs, num_samples=1))

        output_ids.append(token)

        if token in stop_token_ids:
            stopped = True
        else:
            stopped = False

        if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
            output = tokenizer.decode(output_ids, skip_special_tokens=True)
            if stop_str:
                pos = output.rfind(stop_str, l_prompt)
                if pos != -1:
                    output = output[:pos]
                    stopped = True
            yield output

        if stopped:
            break

    del past_key_values


class ChatIO(abc.ABC):
    @abc.abstractmethod
    def prompt_for_input(self, role: str) -> str:
        """Prompt for input from a role."""

    @abc.abstractmethod
    def prompt_for_output(self, role: str) -> None:
        """Prompt for output from a role."""

    @abc.abstractmethod
    def stream_output(self, output_stream: Any, skip_echo_len: int) -> str:
        """Stream output."""


def chat_loop(
    args: Any,
    model_path: str,
    cache_dir: str,
    device: str,
    num_gpus: str,
    max_gpu_memory: str,
    load_8bit: bool,
    conv_template: Optional[str],
    temperature: float,
    max_new_tokens: int,
    chatio: ChatIO,
    debug: bool,
) -> None:
    # Model
    int8 = args.load_8bit
    lora_path = args.lora_weight_path
    model, tokenizer = load_gen_model(
        model_path,
        cache_dir,
        large_model=int8,
        device="cuda:0",
        lora_path=lora_path,
    )

    # Chat
    if conv_template:
        conv = conv_templates[conv_template].copy()
    else:
        conv = get_default_conv_template("vicuna_character").copy()

    # import pdb; pdb.set_trace()
    def chat() -> None:
        while True:
            try:
                inp = chatio.prompt_for_input(conv.roles[0])
            except EOFError:
                inp = ""
            if not inp:
                print("exit...")
                break
            # import pdb; pdb.set_trace()
            conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], "")

            # if is_chatglm:
            #     prompt = conv.messages[conv.offset :]
            #     generate_stream_func = chatglm_generate_stream
            # else:
            generate_stream_func = generate_stream
            prompt = conv.get_prompt()

            skip_echo_len = compute_skip_echo_len(conv, prompt)
            stop_str = (
                conv.sep
                if conv.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.BAIZE]
                else None
            )

            params = {
                "model": model_path,
                "prompt": prompt,
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "stop": stop_str,
            }

            chatio.prompt_for_output(conv.roles[1])
            output_stream = generate_stream_func(model, tokenizer, params, device)
            outputs = chatio.stream_output(output_stream, skip_echo_len)
            # NOTE: strip is important to align with the training data.
            conv.messages[-1][-1] = outputs.strip()

            if debug:
                print("\n", {"prompt": prompt, "outputs": outputs}, "\n")

    while True:
        try:
            conv = get_default_conv_template("vicuna_character").copy()
            chat()
        except KeyboardInterrupt:
            print("restart")
            import pdb

            pdb.set_trace()
            # x = input("input state")
            # if x == "exit":
            #     break
            # else:
            #     continue
