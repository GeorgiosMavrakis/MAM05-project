"""
Streams embeddings from chunks_with_embeddings.jsonl to Qdrant database.
Memory-efficient batch upload.
"""

import json
import time
from pathlib import Path
from typing import Iterator, Dict, List
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams

# ============================================
# CONFIGURATION
# ============================================

QDRANT_URL = "http://134.98.133.84:6333"
QDRANT_API_KEY = None

COLLECTION_NAME = "drug_embeddings"
INPUT_FILE = "chunks_with_embeddings.jsonl"

BATCH_SIZE = 100
VECTOR_SIZE = 384


# ============================================
# HELPERS
# ============================================

def count_lines(filepath: str) -> int:
    """Count lines without loading file."""
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for _ in f:
            count += 1
    return count


def stream_jsonl(filepath: str) -> Iterator[Dict]:
    """Stream JSONL line by line."""
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


# ============================================
# QDRANT SETUP
# ============================================

def create_collection(client: QdrantClient, collection_name: str, vector_size: int) -> None:
    """Create collection if it doesn't exist."""

    # Check if collection exists
    collections = client.get_collections().collections
    exists = any(col.name == collection_name for col in collections)

    if exists:
        print(f"⚠️  Collection '{collection_name}' already exists")
        overwrite = input("Delete and recreate? (y/n): ").lower().strip()
        if overwrite == 'y':
            client.delete_collection(collection_name)
            print(f"✓ Deleted existing collection")
        else:
            print("Using existing collection")
            return

    # Create new collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE  # Cosine similarity (matches your normalized embeddings)
        )
    )
    print(f"✓ Created collection '{collection_name}'")


# ============================================
# UPLOAD
# ============================================

def upload_to_qdrant(
        input_file: str = INPUT_FILE,
        qdrant_url: str = QDRANT_URL,
        api_key: str = QDRANT_API_KEY,
        collection_name: str = COLLECTION_NAME,
        batch_size: int = BATCH_SIZE,
        vector_size: int = VECTOR_SIZE
) -> int:
    """
    Upload embeddings to Qdrant in batches.
    """

    print("\n" + "=" * 60)
    print("UPLOADING TO QDRANT")
    print("=" * 60)

    # Connect to Qdrant
    print(f"\nConnecting to Qdrant at {qdrant_url}...")
    client = QdrantClient(
        url=qdrant_url,
        api_key=api_key
    )

    # Test connection
    try:
        collections = client.get_collections()
        print("✓ Connected to Qdrant")
    except Exception as e:
        print(f"❌ Failed to connect: {e}")
        print("\nTroubleshooting:")
        print("1. Check your VM's public IP is correct")
        print("2. Verify port 6333 is open in Oracle Security List")
        print("3. Ensure Qdrant container is running: docker ps")
        return 0

    # Create/verify collection
    print(f"\nSetting up collection '{collection_name}'...")
    create_collection(client, collection_name, vector_size)

    # Count total
    total_chunks = count_lines(input_file)
    print(f"\nTotal chunks to upload: {total_chunks:,}")
    print(f"Batch size: {batch_size}\n")

    # Upload in batches
    uploaded = 0
    batch_points = []
    start_time = time.time()

    with tqdm(total=total_chunks, desc="Uploading") as pbar:

        for idx, chunk in enumerate(stream_jsonl(input_file)):

            # Create Qdrant point
            point = PointStruct(
                id=idx,  # Sequential ID
                vector=chunk["embedding"],
                payload={
                    "text": chunk["text"],
                    "chunk_index": chunk["chunk_index"],
                    "metadata": chunk["metadata"]
                }
            )

            batch_points.append(point)

            # Upload when batch is full
            if len(batch_points) >= batch_size:
                client.upsert(
                    collection_name=collection_name,
                    points=batch_points
                )
                uploaded += len(batch_points)
                pbar.update(len(batch_points))
                batch_points = []

        # Upload remaining
        if batch_points:
            client.upsert(
                collection_name=collection_name,
                points=batch_points
            )
            uploaded += len(batch_points)
            pbar.update(len(batch_points))

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("UPLOAD COMPLETE")
    print("=" * 60)
    print(f"✓ Uploaded: {uploaded:,} points")
    print(f"✓ Collection: {collection_name}")
    print(f"✓ Time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"✓ Speed: {uploaded / elapsed:.0f} points/sec")

    # Verify
    collection_info = client.get_collection(collection_name)
    print(f"\n✓ Qdrant reports {collection_info.points_count:,} points in collection")

    return uploaded


# ============================================
# VERIFICATION
# ============================================

def test_search(
        qdrant_url: str = QDRANT_URL,
        api_key: str = QDRANT_API_KEY,
        collection_name: str = COLLECTION_NAME
) -> None:
    """Test search functionality."""

    print("\n" + "=" * 60)
    print("TESTING SEARCH")
    print("=" * 60)

    client = QdrantClient(url=qdrant_url, api_key=api_key)

    # Get a random point to use as query
    points = client.scroll(
        collection_name=collection_name,
        limit=1
    )[0]

    if not points:
        print("❌ No points found in collection")
        return

    test_vector = points[0].vector
    test_text = points[0].payload.get("text", "")[:100]

    print(f"\nUsing test query: '{test_text}...'")

    # Search
    results = client.search(
        collection_name=collection_name,
        query_vector=test_vector,
        limit=3
    )

    print(f"\nTop 3 results:")
    for i, result in enumerate(results, 1):
        drug = result.payload["metadata"]["drug_name_brand"]
        text = result.payload["text"][:100]
        print(f"\n{i}. Score: {result.score:.4f}")
        print(f"   Drug: {drug}")
        print(f"   Text: {text}...")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("=" * 60)
    print("QDRANT UPLOAD SCRIPT")
    print("=" * 60)

    print(f"\nConfiguration:")
    print(f"  Qdrant URL: {QDRANT_URL}")
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  Input file: {INPUT_FILE}")
    print(f"  Vector size: {VECTOR_SIZE}")

    if not Path(INPUT_FILE).exists():
        print(f"\n❌ Error: {INPUT_FILE} not found")
        print("Run embedder.py first to generate embeddings")
        exit(1)

    proceed = input("\nProceed with upload? (y/n): ").lower().strip()
    if proceed != 'y':
        exit(0)

    # Upload
    uploaded = upload_to_qdrant()

    if uploaded > 0:
        # Test search
        test = input("\nTest search functionality? (y/n): ").lower().strip()
        if test == 'y':
            test_search()

    print("\n✓ Done! Your Qdrant database is ready")
    print(f"\nDashboard: http://{QDRANT_URL.split('//')[1]}/dashboard")