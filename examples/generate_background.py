import json

from sotopia.generation_utils.generate import generate_background

background = generate_background(
    model_name="gpt-3.5-turbo",
    participants="Jack, Rose",
    topic="borrow money",
    extra_info="Jack speaks first, Rose speaks second",
)

json_file = "data/background.json"

with open(json_file, "w") as f:
    background_dict = json.loads(background.json())
    json.dump(background_dict, f, indent=4)
