import json
import re
from typing import Generic, Type, TypeVar, Optional
from pydantic import BaseModel, Field
import json_repair

OutputType = TypeVar("OutputType", bound=object)
T = TypeVar("T", bound=BaseModel)


class EnvResponse(BaseModel):
    reasoning: str = Field(
        description="first reiterate agents' social goals and then reason about what agents say/do and whether that aligns with their goals."
    )
    p1_rate: int = Field(description="rating of participant 1, on the scale of 0 to 9")
    p2_rate: int = Field(description="rating of participant 2, on the scale of 0 to 9")


class OutputParser(BaseModel, Generic[OutputType]):
    def parse(self, result: str) -> OutputType:
        raise NotImplementedError

    def get_format_instructions(self) -> str:
        raise NotImplementedError


class PydanticOutputParser(OutputParser[T], Generic[T]):
    pydantic_object: Type[T]

    def parse(self, result: str) -> T:
        # Strip markdown code blocks if present
        result = result.strip()
        # Remove the ```json and ``` if both are present
        result = re.sub(r"^```json\s*", "", result).strip(" \n")

        json_result = json_repair.loads(result)
        assert isinstance(json_result, dict)
        if "properties" in json_result:
            return self.pydantic_object.model_validate_json(
                json.dumps(json_result["properties"])
            )
        else:
            parsed_result = self.pydantic_object.model_validate_json(result)
            return parsed_result

    def get_format_instructions(self) -> str:
        return json.dumps(self.pydantic_object.model_json_schema())


class EnvResponsePydanticOutputParser(PydanticOutputParser[EnvResponse]):
    def __init__(self, pydantic_object: Type[EnvResponse] = EnvResponse) -> None:
        super(EnvResponsePydanticOutputParser, self).__init__(
            pydantic_object=pydantic_object
        )

    def parse(self, text: str) -> EnvResponse:
        # remove trailing commas before ) or ] from text
        text = re.sub(r",\s*(\)|\])", r"\1", text)
        response = super().parse(text)
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
