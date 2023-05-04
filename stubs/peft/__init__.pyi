from typing import Any

import torch
from transformers.utils.hub import PushToHubMixin

class PeftModel(PushToHubMixin, torch.nn.Module): ...  # type: ignore[misc]

def set_peft_model_state_dict(
    model: PeftModel, peft_model_state_dict: dict[str, Any]
) -> PeftModel: ...
def get_peft_model(model: PeftModel, peft_config: Any) -> PeftModel: ...
