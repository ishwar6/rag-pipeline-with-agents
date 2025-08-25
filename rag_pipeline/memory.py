from collections import deque
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, List


@dataclass
class ConversationMemory:
    """Persistent storage for recent conversation context.

    Parameters
    ----------
    path: Path, optional
        Filesystem location used to store serialized messages.
    size: int, optional
        Maximum number of recent exchanges to retain.
    """

    path: Path = Path("conversation.json")
    size: int = 5
    messages: deque = field(init=False)

    def __post_init__(self) -> None:
        self.messages = deque(maxlen=self.size)
        if self.path.exists():
            self.messages.extend(json.loads(self.path.read_text()))

    def add(self, role: str, content: str) -> None:
        """Append a message to the memory.

        Parameters
        ----------
        role: str
            Message author, typically ``user`` or ``assistant``.
        content: str
            Textual content of the message.

        Returns
        -------
        None
            Method performs side effects only.
        """
        self.messages.append({"role": role, "content": content})
        self.path.write_text(json.dumps(list(self.messages)))

    def get(self) -> List[Dict[str, str]]:
        """Return the stored conversation history.

        Returns
        -------
        list of dict
            Sequence of messages with roles and content.
        """
        return list(self.messages)
