from typing import Any, TypeVar

import torch

Model = TypeVar("Model", bound=PushToHubMixin)

class PushToHubMixin:
    @classmethod
    def from_pretrained(cls: type[Model], *args: Any, **kwargs: Any) -> Model: ...

class PeftModel(PushToHubMixin, torch.nn.Module): ...

def set_peft_model_state_dict(
    model: PeftModel, peft_model_state_dict: dict[str, Any]
) -> PeftModel: ...
def get_peft_model(model: PeftModel, peft_config: Any) -> PeftModel: ...
