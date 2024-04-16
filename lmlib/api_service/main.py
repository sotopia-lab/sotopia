import transformers
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from lmlib.api_service.chat_model import ChatMessage, ChatModel
from lmlib.api_service.setup_args import setup_inf_args
from lmlib.arguments import InferenceArguments, LoraArguments
from lmlib.serve.lm_inference import load_gen_model

parser = transformers.HfArgumentParser((InferenceArguments, LoraArguments))  # type: ignore
inf_args, lora_args = setup_inf_args()


# hard code model param
model, tokenizer = load_gen_model(
    inf_args.model_path,
    inf_args.cache_dir,
    large_model=inf_args.load_8bit,
    device="cuda:0",
    lora_path=lora_args.lora_weight_path,
)
chat = ChatModel(model, tokenizer, "cuda", 2048, 2)

app = FastAPI()


@app.post("/message/")
async def gen_from_msg(msg: ChatMessage) -> StreamingResponse:
    return StreamingResponse(chat.chat_oneshot(msg))


# for later use, current post method is enough
# @app.put("/message/{msg_id}")
# async def gen_from_id_msg(msg_id: int, msg: ChatMessage):
#     # save id
#     print(f"id: {msg_id}")
#     return StreamingResponse(chat.chat_oneshot(msg))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, port=8000, log_level="info")
