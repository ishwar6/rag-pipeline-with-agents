from collections import deque
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class ConversationMemory:
    """Persistent storage for recent conversation context."""
    path: Path = Path("conversation.json")
    size: int = 5
    messages: deque = field(init=False)

    def __post_init__(self):
        self.messages = deque(maxlen=self.size)
        if self.path.exists():
            self.messages.extend(json.loads(self.path.read_text()))

    def add(self, role, content):
        """Append a message to the memory."""
        self.messages.append({"role": role, "content": content})
        self.path.write_text(json.dumps(list(self.messages)))

    def get(self):
        """Return the stored conversation history."""
        return list(self.messages)
