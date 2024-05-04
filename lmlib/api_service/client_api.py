import json
import os.path as osp
from typing import Any, Dict

import requests

from lmlib.api_service.chat_model import ChatMessage


# write an api service to get the response from the server
class ChatCharacterGPT:
    def __init__(self, public_url: str) -> None:
        self.public_url = public_url

    def check_msg(self, message: Dict[str, Any]) -> bool:
        try:
            ChatMessage.parse_raw(json.dumps(message))
        except Exception as e:
            print(e)
            print("Invalid message type")
            return False
        return True

    def generate_stream(self, message: Dict[str, Any]) -> Any:
        if not self.check_msg(message):
            return
        post_url = osp.join(self.public_url, "message")
        with requests.post(post_url, json=message, stream=True) as response:
            # Set the chunk size to 1024 bytes
            for chunk in response.iter_content(chunk_size=1024):
                yield chunk.decode("utf-8")

    def generate(self, message: Dict[str, Any]) -> str | None:
        if not self.check_msg(message):
            return None
        post_url = osp.join(self.public_url, "message")
        with requests.post(post_url, json=message, stream=True) as response:
            # Set the chunk size to 1024 bytes
            res = ""
            for chunk in response.iter_content(chunk_size=1024):
                res += chunk.decode("utf-8")
        return res


# # post method
# public_url='https://a986-128-2-205-154.ngrok-free.app/message/'
# #url = 'http://127.0.0.1:8000/message/'
# with requests.post(public_url, json=js, stream=True) as response:
#     # Set the chunk size to 1024 bytes
#     for chunk in response.iter_content(chunk_size=100):
#         # Print each chunk to the console
#         print(chunk)

if __name__ == "__main__":
    js = {
        "system": "My name is Eli Dawson, and I'm a 52-year-old forensic psychiatrist. I would like to answer any questions about myself.",
        "messages": [
            ["INTERVIEWER", "Hello"],
            ["ME", "Thank you for having me. It's a pleasure to be here."],
        ],
        "input": "what is your job?",
    }
    public_url = "https://a986-128-2-205-154.ngrok-free.app"
    chat = ChatCharacterGPT(public_url)
    for chunk in chat.generate_stream(js):
        print(chunk)
    print(chat.generate(js))
    # for chunk in chat.generate(js):
    #     print(chunk, end='')
