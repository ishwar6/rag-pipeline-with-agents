from dataclasses import dataclass
from pathlib import Path
from chromadb import Client
from chromadb.config import Settings
from langchain.embeddings import OpenAIEmbeddings


@dataclass
class EmbeddingConfig:
    """Configuration container for embedding models.

    Parameters
    ----------
    model: str
        Name of the embedding model exposed by the provider.
    """

    model: str = "text-embedding-3-small"

    def create(self) -> OpenAIEmbeddings:
        """Instantiate the embedding model.

        Returns
        -------
        OpenAIEmbeddings
            Configured embedding model instance.
        """
        return OpenAIEmbeddings(model=self.model)


@dataclass
class VectorStoreConfig:
    """Configuration for persistent vector storage.

    Parameters
    ----------
    persist_directory: str
        Filesystem path where Chroma will store its database.
    collection_name: str
        Identifier for the document collection.
    """

    persist_directory: str = "vector_db"
    collection_name: str = "documents"

    def create(self, embedding: OpenAIEmbeddings):
        """Initialize or retrieve the Chroma collection.

        Parameters
        ----------
        embedding: OpenAIEmbeddings
            Embedding function used for indexing and similarity search.

        Returns
        -------
        Any
            Chroma collection instance tied to the configured directory and name.
        """
        client = Client(Settings(persist_directory=str(Path(self.persist_directory))))
        return client.get_or_create_collection(name=self.collection_name, embedding_function=embedding)
