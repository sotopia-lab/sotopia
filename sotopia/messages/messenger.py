from .message_classes import Message


class MessengerMixin(object):
    def __init__(self) -> None:
        self.inbox: list[tuple[str, Message]] = []
        """
        Public transcript visible to all other clients
        """

        self.private_inbox: list[tuple[str, Message]] = []
        """
        Private messages keyed by sender; only visible to this agent
        """

    def reset_inbox(self) -> None:
        self.inbox = []
        self.private_inbox = []

    def recv_message(
        self, source: str, message: Message, private: bool = False
    ) -> None:
        if private:
            self.private_inbox.append((source, message))
        else:
            self.inbox.append((source, message))
