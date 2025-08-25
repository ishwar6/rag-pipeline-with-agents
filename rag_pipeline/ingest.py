from typing import Any, Dict, List


class DocumentIngestor:
    """Adds documents to the vector store with optional metadata.

    Parameters
    ----------
    collection: Any
        Chroma collection interface used for insertion.
    """

    def __init__(self, collection: Any) -> None:
        self.collection = collection

    def add(self, texts: List[str], metadatas: List[Dict[str, Any]]) -> None:
        """Insert documents into the collection.

        Parameters
        ----------
        texts: list of str
            Raw document strings to embed and store.
        metadatas: list of dict
            Metadata entries aligned with each text.

        Returns
        -------
        None
            Method performs side effects only.
        """
        ids = [str(i) for i in range(len(texts))]
        self.collection.add(documents=texts, metadatas=metadatas, ids=ids)

