"""XML Renderer for background, goal, observation, etc.
The message passed to the renderer is a string of xml.
If the xml string is not wrapped with <root></root>, we will wrap it with <root></root> automatically.
The tags to render are specified in RenderContext.tags_to_render. ('root' and 'p' are always rendered)
The viewer is specified in RenderContext.viewer. ('human', 'agent_0', 'agent_1', 'environment')
environment: render all text
human: render the raw xml
agent_i: render the text that is viewable by agent_i
"""
from typing import cast

from beartype.door import is_bearable
from lxml import etree

from .base import BaseRenderer, RenderContext


def _render_xml(xml_node: etree._Element | str, context: RenderContext) -> str:
    if isinstance(xml_node, str):
        return xml_node
    else:
        if (
            xml_node.tag in ["root", "p"]
            or xml_node.tag in context.tags_to_render
        ):
            if context.viewer.startswith("agent_"):
                # For each agent, we only render the messages viewable by that agent
                all_visible_children = xml_node.xpath(
                    f"./node()[@viewer='{context.viewer}'] | ./node()[not(@viewer)]"
                )
                assert is_bearable(
                    all_visible_children, list[etree._Element | str]
                )
                cast(list[etree._Element | str], all_visible_children)
                return "".join(
                    _render_xml(child, context) for child in all_visible_children  # type: ignore[attr-defined]
                )
            elif context.viewer == "human":
                # For human, we render the raw xml
                return etree.tostring(xml_node, pretty_print=True).decode(
                    "utf-8"
                )
            elif context.viewer == "environment":
                # For environment, we render all text
                all_text = xml_node.xpath("//text()")
                return "".join(cast(list[str], all_text))
        # Add return statement for the case where none of the conditions are met
        return ""


class XMLRenderer(BaseRenderer):
    def __init__(self) -> None:
        super().__init__()

    def __call__(
        self, xml_string: str, context: RenderContext = RenderContext()
    ) -> str:
        try:
            root = etree.fromstring(xml_string)
        except etree.XMLSyntaxError:
            # try wrapping the xml_string with a pair of root tags
            root = etree.fromstring(f"<root>{xml_string}</root>")

        return _render_xml(root, context)
