import chromadb
from typing import List, Dict

def get_chroma_client(persist_dir: str):
    client = chromadb.PersistentClient(persist_dir)
    return client

def upsert_to_chroma(client, collection_name: str, ids: List[str], docs: List[str], metadatas: List[Dict], embeddings: List[List[float]]):
    # create collection if not exists
    collection = client.get_or_create_collection(name=collection_name)
    print("Upserting %d documents into Chromadb collection '%s'", len(docs), collection_name)
    collection.add(
        ids=ids,
        documents=docs,
        metadatas=metadatas,
        embeddings=embeddings
    )
    return collection


def search_chroma(client, collection_name: str, query_embedding: List[float], k: int = 5):
    collection = client.get_collection(name=collection_name)
    res = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"]
    )
    # result format: dictionaries containing lists for each query (we only have 1 query)
    doc_ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metadatas = res.get("metadatas", [[]])[0]
    distances = res.get("distances", [[]])[0]
    results = []
    for i, doc in enumerate(docs):
        results.append({
            "doc_id": doc_ids[i] if i < len(doc_ids) else None,
            "document": doc,
            "metadata": metadatas[i] if i < len(metadatas) else {},
            "relevance_score": float(distances[i]) if i < len(distances) else None
        })
    return results
