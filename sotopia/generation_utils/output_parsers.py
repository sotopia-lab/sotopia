import json
import re
from typing import Generic, Type, TypeVar, Optional, Any
from pydantic import BaseModel, Field
import json_repair

from sotopia.database import LLMBaseModel
from sotopia.messages import AgentAction

OutputType = TypeVar("OutputType", bound=object)
T = TypeVar("T", bound=BaseModel)


class EnvResponse(LLMBaseModel):
    reasoning: str = Field(
        description="first reiterate agents' social goals and then reason about what agents say/do and whether that aligns with their goals."
    )
    p1_rate: int = Field(description="rating of participant 1, on the scale of 0 to 9")
    p2_rate: int = Field(description="rating of participant 2, on the scale of 0 to 9")


class OutputParser(LLMBaseModel, Generic[OutputType]):
    def parse(self, result: str) -> OutputType:
        raise NotImplementedError

    def get_format_instructions(self) -> str:
        raise NotImplementedError


class PydanticOutputParser(OutputParser[T], Generic[T]):
    pydantic_object: Type[T]

    def parse(self, result: str, context: dict[str, Any] | None = None) -> T:
        # Strip markdown code blocks if present
        result = result.strip()
        # Remove the ```json and ``` if both are present
        result = re.sub(r"^```json\s*", "", result).strip(" \n")

        json_result = json_repair.loads(result)
        assert isinstance(json_result, dict)

        # Handle nested type-value structure
        def extract_value(obj: dict[str, Any] | list[Any] | str) -> Any:
            if isinstance(obj, dict):
                if "value" in obj:
                    return obj["value"]
                return {k: extract_value(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [extract_value(item) for item in obj]
            return obj

        json_result = extract_value(json_result)
        if isinstance(json_result, dict) and "properties" in json_result:
            return self.pydantic_object.model_validate_json(
                json.dumps(json_result["properties"])
            )
        else:
            data = json_result

        # Best-effort normalization for common non-canonical agent outputs.
        # Some LLMs emit verbs like "kill", "vote", etc. as top-level
        # `action_type` values. Normalize them to the canonical schema
        # expected by `AgentAction` to avoid validation failures.
        try:
            if self.pydantic_object is AgentAction and isinstance(data, dict):
                at = data.get("action_type")
                # Allowed literals per AgentAction
                valid_types = {"none", "speak", "non-verbal communication", "action", "leave"}
                if isinstance(at, str) and at.strip() not in valid_types:
                    verb_phrase = at.strip()
                    parts = verb_phrase.split()
                    verb = parts[0].lower() if parts else ""
                    # verbs that should be treated as generic 'action'
                    action_verbs = {
                        "kill",
                        "vote",
                        "save",
                        "poison",
                        "inspect",
                        "investigate",
                        "guard",
                        "protect",
                        "heal",
                        "lynch",
                    }
                    if verb in action_verbs:
                        # If argument already exists, prefer it; otherwise try to recover from the verb phrase
                        arg = data.get("argument")
                        if not arg or (isinstance(arg, str) and not arg.strip()):
                            if len(parts) > 1:
                                arg = " ".join(parts[1:])
                            else:
                                # fallback to using the verb itself
                                arg = verb
                        # ensure argument is a string
                        if not isinstance(arg, str):
                            arg = str(arg)
                        data["action_type"] = "action"
                        data["argument"] = f"{verb} {arg}" if arg else verb
                    else:
                        # map informal speak verbs to 'speak'
                        if verb in {"say", "says", "speak", "talk"}:
                            data["action_type"] = "speak"
                            if "argument" not in data or not data.get("argument"):
                                # recover remaining phrase if present
                                data["argument"] = " ".join(parts[1:]) if len(parts) > 1 else ""
        except Exception:
            # Never fail normalization; fall back to original data and let pydantic raise if needed
            pass

        # Use model_validate with context if provided, otherwise use model_validate_json for backward compatibility
        if context is not None:
            return self.pydantic_object.model_validate(data, context=context)
        else:
            # For structured output, use normalized data even without context
            return self.pydantic_object.model_validate(data)

    def get_format_instructions(self) -> str:
        return json.dumps(self.pydantic_object.model_json_schema())


class EnvResponsePydanticOutputParser(PydanticOutputParser[EnvResponse]):
    def __init__(self, pydantic_object: Type[EnvResponse] = EnvResponse) -> None:
        super(EnvResponsePydanticOutputParser, self).__init__(
            pydantic_object=pydantic_object
        )

    def parse(self, text: str, context: dict[str, Any] | None = None) -> EnvResponse:
        # remove trailing commas before ) or ] from text
        text = re.sub(r",\s*(\)|\])", r"\1", text)
        response = super().parse(text, context=context)
        if isinstance(response, EnvResponse):
            return response
        else:
            raise ValueError(f"Expected EnvResponse, got {type(response)}")

    def get_format_instructions(self) -> str:
        format_instruction = super().get_format_instructions()
        return format_instruction


class StrOutputParser(OutputParser[str]):
    def parse(self, result: str) -> str:
        return result

    def get_format_instructions(self) -> str:
        return ""


class ScriptOutputParser(OutputParser[str]):
    def parse(self, result: str) -> str:
        return result

    def get_format_instructions(self) -> str:
        return ""


class ListOfIntOutputParser(OutputParser[list[int]]):
    number_of_int: Optional[int] = None
    range_of_int: Optional[tuple[int, int]] = None

    def __init__(
        self,
        number_of_int: Optional[int] = None,
        range_of_int: Optional[tuple[int, int]] = None,
    ):
        """
        Parse the output to a list of integers

        Args:
            number_of_int (int | None): The number of integers in the output. If None, the number of integers is not fixed.
        """
        super().__init__()
        self.number_of_int = number_of_int
        self.range_of_int = range_of_int

    def _get_description_text(self) -> str:
        return f"a list of{' ' + str(self.number_of_int) if self.number_of_int else ''} intergers{' within the range of' + str(self.range_of_int) if self.range_of_int else ''} separated by spaces. Don't output anything else. Format example: 1 2 3 4 5"

    def get_format_instructions(self) -> str:
        return "Please output " + self._get_description_text()

    def parse(self, output: str) -> list[int]:
        try:
            output_loaded = output.split(" ")
            result = [int(x) for x in output_loaded]
            if self.number_of_int and len(result) != self.number_of_int:
                msg = f"Expect {self.number_of_int} integers, got {len(result)}"
                raise ValueError(msg)
            if self.range_of_int:
                for x in result:
                    if x < self.range_of_int[0] or x > self.range_of_int[1]:
                        msg = f"Expect integers within the range of {self.range_of_int}, got {result}"
                        raise ValueError(msg)
            return result
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            msg = f"Exception {e}: the output format is not correct. Expect {self._get_description_text()}, got {output}"
            raise ValueError(msg)

    @property
    def _type(self) -> str:
        """Return the type key."""
        return "list[int]"
