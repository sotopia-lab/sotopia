"""
Chat with a model with command line interface.
"""

import os
import re
from argparse import Namespace
from typing import Any, List

import transformers
from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown

from lmlib.arguments import InferenceArguments, LoraArguments
from lmlib.serve.lm_inference import ChatIO, chat_loop


class SimpleChatIO(ChatIO):
    def prompt_for_input(self, role: str) -> str:
        return input(f"{role}: ")

    def prompt_for_output(self, role: str) -> None:
        print(f"{role}: ", end="", flush=True)

    def stream_output(self, output_stream: List[Any], skip_echo_len: int) -> str:
        pre = 0
        for outputs in output_stream:
            outputs = outputs[skip_echo_len:].strip()
            outputs = outputs.split(" ")
            now = len(outputs) - 1
            if now > pre:
                print(" ".join(outputs[pre:now]), end=" ", flush=True)
                pre = now
        print(" ".join(outputs[pre:]), flush=True)
        return " ".join(outputs)


class RichChatIO(ChatIO):
    _prompt_session: Any

    def __init__(self) -> None:
        self._prompt_session = PromptSession(history=InMemoryHistory())
        self._completer = WordCompleter(
            words=["!exit", "!reset"], pattern=re.compile("$")
        )
        self._console = Console()

    def prompt_for_input(self, role: str) -> Any:
        self._console.print(f"[bold]{role}:")
        # TODO(suquark): multiline input has some issues. fix it later.
        prompt_input = self._prompt_session.prompt(
            completer=self._completer,
            multiline=False,
            auto_suggest=AutoSuggestFromHistory(),
            key_bindings=None,
        )
        self._console.print()
        return prompt_input

    def prompt_for_output(self, role: str) -> None:
        self._console.print(f"[bold]{role}:")

    def stream_output(self, output_stream: List[Any], skip_echo_len: int) -> Any:
        """Stream output from a role."""
        # TODO(suquark): the console flickers when there is a code block
        #  above it. We need to cut off "live" when a code block is done.

        # Create a Live context for updating the console output
        with Live(console=self._console, refresh_per_second=4) as live:
            # Read lines from the stream
            for outputs in output_stream:
                accumulated_text = outputs[skip_echo_len:]
                if not accumulated_text:
                    continue
                # Render the accumulated text as Markdown
                # NOTE: this is a workaround for the rendering "unstandard markdown"
                #  in rich. The chatbots output treat "\n" as a new line for
                #  better compatibility with real-world text. However, rendering
                #  in markdown would break the format. It is because standard markdown
                #  treat a single "\n" in normal text as a space.
                #  Our workaround is adding two spaces at the end of each line.
                #  This is not a perfect solution, as it would
                #  introduce trailing spaces (only) in code block, but it works well
                #  especially for console output, because in general the console does not
                #  care about trailing spaces.
                lines = []
                for line in accumulated_text.splitlines():
                    lines.append(line)
                    if line.startswith("```"):
                        # Code block marker - do not add trailing spaces, as it would
                        #  break the syntax highlighting
                        lines.append("\n")
                    else:
                        lines.append("  \n")
                markdown = Markdown("".join(lines))
                # Update the Live console output
                live.update(markdown)
        self._console.print()
        return outputs[skip_echo_len:]


def main(args: Namespace) -> None:
    if args.gpus:
        if args.num_gpus and len(args.gpus.split(",")) < int(args.num_gpus):
            raise ValueError(
                f"Larger --num-gpus ({args.num_gpus}) than --gpus {args.gpus}!"
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    if args.style == "simple":
        chatio: Any = SimpleChatIO()
    elif args.style == "rich":
        chatio = RichChatIO()
    else:
        raise ValueError(f"Invalid style for console: {args.style}")
    try:
        chat_loop(
            args,
            args.model_path,
            args.cache_dir,
            args.device,
            args.num_gpus,
            args.max_gpu_memory,
            args.load_8bit,
            args.conv_template,
            args.temperature,
            args.max_new_tokens,
            chatio,
            args.debug,
        )
    except KeyboardInterrupt:
        print("exit...")


if __name__ == "__main__":
    parser = transformers.HfArgumentParser((InferenceArguments, LoraArguments))  # type: ignore
    (
        inf_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()

    args = Namespace(**vars(inf_args), **vars(lora_args))
    main(args)
