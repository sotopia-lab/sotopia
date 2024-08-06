import logging
from typing import Any
from langchain_core.messages import BaseMessage
from langchain.callbacks import StdOutCallbackHandler

logging.addLevelName(15, "LangChain")


class LoggingCallbackHandler(StdOutCallbackHandler):
    """Callback Handler that prints to std out."""

    always_verbose = True

    def __init__(self, name: str) -> None:
        """Initialize callback handler."""
        super().__init__()
        self.logger = logging.getLogger(name)
        self.prompt = ""

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        **kwargs: Any,
    ) -> None:
        self.prompt = str(messages[0][0].content)
        logging.log(15, f"Prompt after formatting:{self.prompt}")

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
        self, error: BaseException | KeyboardInterrupt, **kwargs: Any
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
        # leave only prompt for environment
        pass

    def retrive_prompt(self) -> str:
        return self.prompt

    def on_agent_finish(self, *args: Any, **kwargs: Any) -> None:
        """Run on agent end."""
        pass
