"""
Conversation prompt template.
"""

import dataclasses
from enum import Enum, auto
from typing import Any, List


class SeparatorStyle(Enum):
    """Different separator style."""

    SINGLE = auto()
    TWO = auto()
    DOLLY = auto()
    OASST_PYTHIA = auto()
    BAIZE = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""

    system: str
    roles: list[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str | None = None

    # Used for the state in the gradio servers.
    # TODO(lmzheng): refactor this
    conv_id: Any = None
    skip_next: bool = False
    model_name: str | None = None

    def get_prompt(self) -> str:
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += self.sep + " " + role + ": " + message
                else:
                    ret += self.sep + " " + role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            assert (
                self.sep2 is not None
            ), "sep2 should not be None when using TWO style."
            seps = (self.sep, self.sep2)
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message.strip() + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.DOLLY:
            assert (
                self.sep2 is not None
            ), "sep2 should not be None when using DOLLY style."
            seps = (self.sep, self.sep2)
            ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ":\n" + message + seps[i % 2]
                    if i % 2 == 1:
                        ret += "\n\n"
                else:
                    ret += role + ":\n"
            return ret
        elif self.sep_style == SeparatorStyle.OASST_PYTHIA:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.BAIZE:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += "\n" + role + message
                else:
                    ret += "\n" + role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role: str, message: str) -> None:
        self.messages.append([role, message])

    def to_gradio_chatbot(self) -> list[list[str | None]]:
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self) -> "Conversation":
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id,
            model_name=self.model_name,
        )

    def dict(self) -> dict[str, Any]:
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
            "model_name": self.model_name,
        }


conv_vicuna_v1_1 = Conversation(
    system="A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions.",
    roles=["USER", "ASSISTANT"],
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_emily = Conversation(
    system="My name is Emily Harrison and I am a 32-year-old woman. I am very compassionate and values caring for other people. I would like to answer any questions about myself.",
    roles=["INTERVIEWER", "ME"],
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_sasha = Conversation(
    system="My name is Sasha Ramirez and I am a 42-year-old female police officer. My decision-making style is logical, and my primary values are authority and loyalty. I would like to answer any questions about myself.",
    roles=["INTERVIEWER", "ME"],
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_eli = Conversation(
    system="My name is Eli Dawson, and I'm a 52-year-old forensic psychiatrist. I would like to answer any questions about myself.",
    roles=["INTERVIEWER", "ME"],
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

conv_templates = {
    "vicuna_v1_1": conv_vicuna_v1_1,
    "vicuna_emily": conv_emily,
    "vicuna_sasha": conv_sasha,
    "vicuna_eli": conv_eli,
}


def get_default_conv_template(model_name: str) -> Conversation:
    return conv_vicuna_v1_1


def compute_skip_echo_len(
    conv: Conversation, prompt: str, is_chatglm: bool = False
) -> int:
    if is_chatglm:
        assert len(conv.messages) >= 2
        assert conv.messages[-2][1] is not None
        skip_echo_len = len(conv.messages[-2][1]) + 1
    else:
        skip_echo_len = len(prompt) + 1 - prompt.count("</s>") * 3
    return skip_echo_len


if __name__ == "__main__":
    default_conversation = conv_templates["vicuna_v1.1"]
    print(default_conversation.get_prompt())
