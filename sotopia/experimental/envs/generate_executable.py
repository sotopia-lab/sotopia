from jinja2 import Environment, FileSystemLoader
import os
from typing import Any


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
                    output.append(f"{key} = {str(v)}")
        elif isinstance(val, (str, int, float, bool)):
            output.append(f'{key} = "{str(val)}"')
        else:
            raise ValueError(f"Unsupported type {type(val)}")
    return "\n".join(output)


def generate_executable(input_params: dict[str, Any]) -> str:
    env = Environment(
        loader=FileSystemLoader(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
        )
    )
    env.filters["render_bool"] = render_bool
    template = env.get_template("multiagents.jinja2")
    return template.render(input_params, render_dict=render_dict)
