import logging
from typing import Any

from langchain.callbacks import StdOutCallbackHandler

logging.addLevelName(15, "LangChain")


class LoggingCallbackHandler(StdOutCallbackHandler):
    """Callback Handler that prints to std out."""

    always_verbose = True

    def __init__(self, name: str) -> None:
        """Initialize callback handler."""
        super().__init__()
        self.logger = logging.getLogger(name)

    def on_chain_start(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_chain_end(self, *args: Any, **kwargs: Any) -> None:
        pass

    def on_agent_action(self, *args: Any, **kwargs: Any) -> Any:
        pass

    def on_tool_end(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        pass

    def on_tool_error(
        self, error: Exception | KeyboardInterrupt, **kwargs: Any
    ) -> None:
        """Do nothing."""
        pass

    def on_text(
        self,
        text: str,
        color: str | None = None,
        end: str = "",
        **kwargs: Any,
    ) -> None:
        """Run when agent ends."""
        logging.log(15, f"LLM Call: {text}")

    def on_agent_finish(self, *args: Any, **kwargs: Any) -> None:
        """Run on agent end."""
        pass
