from dataclasses import dataclass
from pathlib import Path
from chromadb import Client
from chromadb.config import Settings
from langchain.embeddings import OpenAIEmbeddings


@dataclass
class EmbeddingConfig:
    """Configuration for embedding models."""
    model: str = "text-embedding-3-small"

    def create(self):
        """Instantiate the embedding model."""
        return OpenAIEmbeddings(model=self.model)


@dataclass
class VectorStoreConfig:
    """Configuration for the vector database."""
    persist_directory: str = "vector_db"
    collection_name: str = "documents"

    def create(self, embedding):
        """Initialize the Chroma vector store."""
        client = Client(Settings(persist_directory=str(Path(self.persist_directory))))
        return client.get_or_create_collection(name=self.collection_name, embedding_function=embedding)
