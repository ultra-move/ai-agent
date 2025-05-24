from datetime import datetime
import chromadb
from chromadb.utils import embedding_functions
import uuid


class VectorStore:
    def __init__(self, storage_path):
        """
        Initializes the vector store with a local persistent client and a default embedding function.

        Args:
            storage_path (str): The file path where the vector store will be persisted.
        """
        self.client = chromadb.PersistentClient(path=storage_path)
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()
        self.cached_collections = {}
        print("Vector Store initialized")

    def heartbeat(self):
        """
        Checks the connection of the local client.

        Returns:
            bool: True if the client is reachable.
        """
        return self.client.heartbeat()

    def cache_collections(self, collection_list):
        """
        Caches each collection from collection_list for improved response times.

        Args:
            collection_list (list): A list of collection names to be cached.
        """
        if not collection_list:
            print("No collections provided to cache.")
            return

        for collection_name in collection_list:
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            self.cached_collections[collection_name] = collection

    def create_collection(self, collection_name):
        """
        Creates a new collection or retrieves it from the cache.

        Args:
            collection_name (str): The name of the collection.

        Returns:
            object: The collection object.
        """
        cached_collection = self.cached_collections.get(collection_name)
        if cached_collection is not None:
            return cached_collection

        collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.embedding_function,
            metadata={
                "description": f"This is the collection containing documents about {collection_name}",
                "created": str(datetime.now())
            })
        self.cached_collections[collection_name] = collection
        return collection


    def get_collection(self, collection_name):
        """
        Retrieves a collection by name.

        Args:
            collection_name (str): Name of the collection.

        Returns:
            object: The collection object.
        """
        cached_collection = self.cached_collections.get(collection_name)
        if cached_collection is not None:
            return cached_collection

        collection = self.client.get_collection(
            name=collection_name,
            embedding_function=self.embedding_function
        )
        self.cached_collections[collection_name] = collection
        return collection

    def add_document(self, collection_name, document, metadata=None, id=None):
        """
        Adds a document to a collection.

        Args:
            collection_name (str): Target collection name.
            document (str): The document text.
            metadata (dict, optional): Metadata for the document.
            id (str, optional): Document ID. Auto-generates if not provided.
        """
        if metadata is None:
            metadata = {}
        if id is None:
            id = str(uuid.uuid4())

        collection = self.get_collection(collection_name)
        collection.add(
            documents=[document],
            metadatas=[metadata],
            ids=[id]
        )
        return id

    def query_collection(self, collection_name, query_text, num_documents=5):
        """
        Queries a single collection.

        Args:
            collection_name (str): Name of the collection.
            query_text (str): The query string.
            num_documents (int): Number of results to return.

        Returns:
            dict: Query results.
        """
        collection = self.get_collection(collection_name)
        return collection.query(
            query_texts=[query_text],
            n_results=num_documents
        )

    def delete_collection(self, collection_name):
        """
        Deletes a collection.

        Args:
            collection_name (str): Name of the collection to delete.
        """
        try:
            del self.cached_collections[collection_name]
        except KeyError:
            print(f"Collection '{collection_name}' not in cache.")
        except Exception as e:
            print(f"Error removing collection from cache: {e}")

        try:
            self.client.delete_collection(name=collection_name)
        except Exception as e:
            print(f"Error deleting collection from client: {e}")

        print(f"Deleted Collection: {collection_name}")

    def list_collections(self):
        """
        Lists all existing collections.

        Returns:
            list: List of collection names.
        """
        return [c.name for c in self.client.list_collections()]
