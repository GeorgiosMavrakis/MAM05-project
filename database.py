import chromadb
import embedder

chroma_client = chromadb.Client(
    settings=chromadb.Settings(
        persist_directory="./chroma_db"
    )
)

collection = chroma_client.get_or_create_collection(
    name="rag_collection"
)

def search_db(query: str, k: int = 3):

    q_embed = embedder.get_embedding(query)

    results = collection.query(
        query_embeddings=[q_embed],
        n_results=k
    )

    return results["documents"][0]
