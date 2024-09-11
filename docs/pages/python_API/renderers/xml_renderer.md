# XML Renderer for Background, Goal, Observation, etc.

This module provides functionality for rendering XML based on different contexts. It includes a helper function `_render_xml` and a class `XMLRenderer`.

## Class: `XMLRenderer`

### Description

`XMLRenderer` is utilized for rendering XML strings in various contexts. It can parse XML strings and handle errors associated with XML syntax.

### Methods

#### `__init__(self) -> None`

Constructor to initialize the `XMLRenderer` object. Sets up the XML parser.

#### `__call__(self, xml_string: str, context: RenderContext = RenderContext()) -> str`

Renders the given XML string according to the provided render context.

- **Parameters:**
  - `xml_string` (str): The XML content to be rendered.
  - `context` (RenderContext, optional): The context which determines how XML should be rendered. Defaults to an empty `RenderContext`.

- **Returns:**
  - `str`: The rendered XML content based on the specified context.

### Helper Function

#### `_render_xml(xml_node: etree._Element | str, context: RenderContext) -> str`

Renders an XML node recursively based on the viewer type specified in the context.

- **Parameters:**
  - `xml_node` (etree._Element | str): The XML element or string content to be rendered.
  - `context` (RenderContext): The context which defines rendering rules.

- **Returns:**
  - `str`: The rendered content as a string.

### Usage Example

```python
from mymodule import XMLRenderer, RenderContext

renderer = XMLRenderer()
context = RenderContext(viewer="human", tags_to_render=["goal", "observation"])

xml_string = """
<root>
    <goal>Achieve world peace</goal>
    <observation viewer="agent_1">Enemy spotted</observation>
    <observation>Clear skies</observation>
</root>
"""

rendered_output = renderer(xml_string, context)
print(rendered_output)
```

In this example, the `XMLRenderer` is instantiated, a rendering context is created, and an XML string is rendered based on the given context.
