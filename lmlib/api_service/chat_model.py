from typing import Any, List

import torch
from pydantic import BaseModel

from lmlib.utils.conversation import (
    Conversation,
    SeparatorStyle,
    compute_skip_echo_len,
)


class ChatMessage(BaseModel):
    system: str
    messages: List[List[str]]  # list of [role, message]
    input: str


def get_conv_from_json(message: ChatMessage) -> Conversation:
    system = message.system
    messages = message.messages
    return Conversation(
        system=system,
        roles=["INTERVIEWER", "ME"],
        messages=messages,
        offset=0,
        sep_style=SeparatorStyle.TWO,
        sep=" ",
        sep2="</s>",
    )


class ChatModel:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str,
        context_len: int = 2048,
        stream_interval: int = 2,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.context_len = context_len
        self.stream_interval = stream_interval

    @torch.inference_mode()
    def generate_stream(
        self,
        params: Any,
        device: str,
        context_len: int = 2048,
        stream_interval: int = 2,
    ) -> Any:
        model = self.model
        tokenizer = self.tokenizer
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
                    out = model(
                        torch.as_tensor([input_ids], device=device),
                        use_cache=True,
                    )
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

    def stream_output(self, output_stream: List[Any], skip_echo_len: int) -> Any:
        pre = 0
        for outputs in output_stream:
            outputs = outputs[skip_echo_len:].strip()
            outputs = outputs.split(" ")
            now = len(outputs) - 1
            if now > pre:
                yield " ".join(outputs[pre:now]) + " "
                pre = now
        yield " ".join(outputs[pre:])

    def prompt_for_input(self, role: str) -> str:
        return input(f"{role}: ")

    def prompt_for_output(self, role: str) -> None:
        print(f"{role}: ", end="", flush=True)

    def chat_oneshot(
        self,
        msg: ChatMessage,
        temperature: float = 0.7,
        max_new_tokens: int = 2048,
    ) -> Any:
        conv = get_conv_from_json(msg)
        inp = msg.input
        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], "")
        prompt = conv.get_prompt()

        skip_echo_len = compute_skip_echo_len(conv, prompt)
        stop_str = (
            conv.sep
            if conv.sep_style in [SeparatorStyle.SINGLE, SeparatorStyle.BAIZE]
            else None
        )

        params = {
            "prompt": prompt,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
            "stop": stop_str,
        }

        # self.prompt_for_output(conv.roles[1])
        outstream = self.generate_stream(params, self.device)
        return self.stream_output(outstream, skip_echo_len)

        # output_stream = self.generate_stream(params, self.device) # self.device is "cuda"
        # return chatio.stream_output(output_stream, skip_echo_len)
