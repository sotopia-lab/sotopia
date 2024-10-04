# RenderContext and BaseRenderer Documentation

## RenderContext

`RenderContext` is a Pydantic model that defines the context required for rendering strings. It includes the following attributes:

### Attributes

- **viewer** (`str`):
  - Description: The viewer of the rendered string.
  - Default: `"human"`
  - Constraints: Must be one of `'human'`, `'environment'`, or `'agent_i'` where `i` is a non-negative integer representing the index of the agent in the episode log.

- **verbose** (`bool`):
  - Description: Whether to render the verbose version of the string.
  - Default: `False`

- **tags_to_render** (`list[str]`):
  - Description: The special tags to render.
  - Default: `[]`

### Validators

- **viewer_must_be_valid**:
  - Ensures that the `viewer` field follows the required constraints.
  - Raises a `ValueError` if `viewer` is not one of the valid options.

### Example

```python
from pydantic import ValidationError

try:
    context = RenderContext(viewer="agent_1", verbose=True, tags_to_render=["tag1", "tag2"])
except ValidationError as e:
    print(e)
```

## BaseRenderer

`BaseRenderer` is a base class that represents a simple string renderer.

### Methods

- **`__call__(self, input_string: str, context: RenderContext) -> str`**:
  - Description: Renders the given `input_string` based on the provided `context`. Currently, it simply returns the `input_string` without any modification.
  - Parameters:
    - **input_string** (`str`): The string to be rendered.
    - **context** (`RenderContext`): The rendering context that includes viewer, verbosity, and tags.
  - Returns: The input string (`str`).

### Example

```python
context = RenderContext(viewer="human", verbose=False, tags_to_render=["highlight"])
renderer = BaseRenderer()
rendered_string = renderer("This is a test string.", context)
print(rendered_string)  # Output: This is a test string.
```
