from jinja2 import Environment, FileSystemLoader
import json
from argparse import Namespace
from argparse import ArgumentParser
import os
from typing import Any


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str, default="output.toml")
    return parser.parse_args()


def render_bool(
    value: str,
) -> str:  # the default jinja2 output of boolean is capitalized, however we want it to be lowercase for toml
    return ("true" if value else "false") if isinstance(value, bool) else value


def render_dict(
    value: dict[str, Any], prefix: str = ""
) -> str:  # a function for rendering dict into toml format
    if not isinstance(value, dict):
        raise ValueError("value must be a dictionary")
    output = []
    for key, val in value.items():
        if isinstance(val, dict):
            output.append("[{}]".format(f"{prefix}.{key}"))
            output.append(render_dict(val, f"{prefix}.{key}"))
        elif isinstance(val, list):
            for v in val:
                if isinstance(v, dict):
                    output.append("[[{}]]".format(f"{prefix}.{key}"))
                    output.append(render_dict(v, f"{prefix}.{key}."))
                else:
                    output.append(f"{key} = {str(v).lower()}")
        elif isinstance(val, (str, int, float, bool)):
            output.append(f'{key} = "{str(val).lower()}"')
        else:
            raise ValueError(f"Unsupported type {type(val)}")
    return "\n".join(output)


if __name__ == "__main__":
    args = parse_args()
    # join with absolute current path
    env = Environment(
        loader=FileSystemLoader(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        )
    )
    env.filters["render_bool"] = render_bool
    template = env.get_template("multiagents.jinja2")

    # Load JSON instead of TOML
    with open(args.input, "r") as f:
        input_data = json.load(f)

    with open(args.output, "w") as f:
        f.write(template.render(input_data, render_dict=render_dict))
