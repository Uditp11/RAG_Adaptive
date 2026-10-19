# Switching the vector store to Qdrant

By default the app stores document chunks in an in-memory FAISS index, which is lost on restart. The Qdrant version of the same code is still in `src/rag/retriever_setup.py`, commented out. This page lists what to change to use it.

## 1. Get a Qdrant instance

Either run a local Qdrant server (default port 6333) or create a cluster on Qdrant Cloud and create an API key for it. You need the cluster URL and, for the cloud, the API key.

## 2. Set the environment variables

Add these to `.env`. They are read in `src/core/config.py`:

```env
QDRANT_URL=https://your-cluster-url:6333
QDRANT_API_KEY=your_qdrant_key
QDRANT_CODE_COLLECTION=codebase
QDRANT_DOCS_COLLECTION=guidelines
```

The commented code writes to the collection named by `QDRANT_CODE_COLLECTION`.

## 3. Check the connection

`langchain-qdrant` is already in `requirements.txt` and installs `qdrant-client`.

```python
from qdrant_client import QdrantClient

client = QdrantClient(url="https://your-cluster-url:6333", api_key="your_qdrant_key")
print(client.get_collections())
```

## 4. Swap the code in `src/rag/retriever_setup.py`

1. Uncomment `from langchain_qdrant import QdrantVectorStore`.
2. In `retriever_chain()`, uncomment the `QdrantVectorStore.from_documents(...)` call and comment out the `FAISS.from_documents(...)` call.
3. In `get_retriever()`, uncomment the Qdrant block that builds `vectorstore` and `retriever`, and comment out the FAISS branch that uses `_faiss_vectorstore` and the placeholder document.

Restart the API, upload a document and ask a question about it to check that retrieval works.
