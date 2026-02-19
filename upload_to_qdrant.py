"""
Streams embeddings from chunks_with_embeddings.jsonl to Qdrant database.
Optimized for large files (50GB+) with parallel uploads and fast parsing.
"""

import json
import time
import threading
import queue
import orjson
from pathlib import Path
from typing import Iterator, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams

try:
    # Faster JSON parsing
    json_loads = orjson.loads
except ImportError:
    json_loads = json.loads

# ============================================
# CONFIGURATION
# ============================================

QDRANT_URL = "http://134.98.133.84:6333"
QDRANT_API_KEY = None

COLLECTION_NAME = "drug_embeddings"
INPUT_FILE = "chunks_with_embeddings_whole.jsonl"

# Optimized for large files
BATCH_SIZE = 2000  # Increased from 100 for better throughput
NUM_UPLOAD_THREADS = 6  # Parallel upload threads
VECTOR_SIZE = 384
SKIP_LINE_COUNT = True  # Skip counting for huge files (use estimate)


# ============================================
# HELPERS
# ============================================

def estimate_line_count(filepath: str, sample_lines: int = 1000) -> Tuple[int, bool]:
    """Estimate total lines by sampling first N lines."""
    if not SKIP_LINE_COUNT:
        count = 0
        with open(filepath, 'r', encoding='utf-8') as f:
            for _ in f:
                count += 1
        return count, False

    # For huge files, estimate based on file size and sample line size
    sample_size = 0
    sample_count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for _ in range(sample_lines):
            line = f.readline()
            if not line:
                break
            sample_size += len(line.encode('utf-8'))
            sample_count += 1

    if sample_count == 0:
        return 1000, True  # Default estimate

    file_size = Path(filepath).stat().st_size
    avg_line_size = sample_size / sample_count
    estimated = int(file_size / avg_line_size)
    print(f"  Estimated {estimated:,} lines (sampled {sample_count} lines, avg {avg_line_size:.0f} bytes)")
    return estimated, True


def stream_jsonl(filepath: str, use_fast_parser: bool = True) -> Iterator[Dict]:
    """Stream JSONL line by line with fast parser."""
    parser = json_loads if use_fast_parser else json.loads
    with open(filepath, 'r', encoding='utf-8', buffering=65536) as f:
        for line in f:
            if line.strip():
                yield parser(line)


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

def upload_batch_worker(
        client: QdrantClient,
        collection_name: str,
        batch_queue: queue.Queue,
        pbar: tqdm,
        results: List
) -> None:
    """Worker thread that uploads batches from queue."""
    while True:
        try:
            item = batch_queue.get(timeout=1)
            if item is None:  # Poison pill
                break

            batch_points, batch_len = item

            # Upload with retry logic
            retries = 3
            while retries > 0:
                try:
                    client.upsert(
                        collection_name=collection_name,
                        points=batch_points
                    )
                    results.append(batch_len)
                    pbar.update(batch_len)
                    break
                except Exception as e:
                    retries -= 1
                    if retries > 0:
                        time.sleep(2)  # Wait before retry
                    else:
                        print(f"\n❌ Failed to upload batch after retries: {e}")
                        raise

            batch_queue.task_done()
        except queue.Empty:
            continue


def upload_to_qdrant(
        input_file: str = INPUT_FILE,
        qdrant_url: str = QDRANT_URL,
        api_key: str = QDRANT_API_KEY,
        collection_name: str = COLLECTION_NAME,
        batch_size: int = BATCH_SIZE,
        vector_size: int = VECTOR_SIZE,
        num_threads: int = NUM_UPLOAD_THREADS
) -> int:
    """
    Upload embeddings to Qdrant in parallel batches.
    """

    print("\n" + "=" * 60)
    print("UPLOADING TO QDRANT (OPTIMIZED)")
    print("=" * 60)

    # Connect to Qdrant
    print(f"\nConnecting to Qdrant at {qdrant_url}...")
    client = QdrantClient(
        url=qdrant_url,
        api_key=api_key,
        timeout=60.0  # Increased timeout for large batches
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

    # Estimate total
    print(f"\nAnalyzing file size...")
    total_chunks, is_estimate = estimate_line_count(input_file)
    print(f"✓ Total chunks to upload: {total_chunks:,}{' (estimated)' if is_estimate else ''}")
    print(f"✓ Batch size: {batch_size}")
    print(f"✓ Upload threads: {num_threads}\n")

    # Upload in parallel batches
    uploaded = 0
    batch_points = []
    batch_queue = queue.Queue(maxsize=num_threads * 2)
    results = []
    start_time = time.time()

    # Start worker threads
    executor = ThreadPoolExecutor(max_workers=num_threads)
    futures = []
    upload_pbar = tqdm(
        total=None if is_estimate else total_chunks,
        desc="Uploaded",
        position=0
    )
    for _ in range(num_threads):
        future = executor.submit(
            upload_batch_worker,
            client,
            collection_name,
            batch_queue,
            upload_pbar,
            results
        )
        futures.append(future)

    # Queue batches from stream
    try:
        for idx, chunk in enumerate(stream_jsonl(input_file)):

            # Create Qdrant point (minimize payload size)
            point = PointStruct(
                id=idx,
                vector=chunk["embedding"],
                payload={
                    "text": chunk["text"],
                    "chunk_index": chunk.get("chunk_index", idx),
                    "metadata": chunk.get("metadata", {})
                }
            )

            batch_points.append(point)

            # Queue when batch is full
            if len(batch_points) >= batch_size:
                batch_queue.put((batch_points, len(batch_points)))
                batch_points = []

        # Queue remaining
        if batch_points:
            batch_queue.put((batch_points, len(batch_points)))

    except Exception as e:
        print(f"\n❌ Error during upload: {e}")
        # Stop workers
        for _ in range(num_threads):
            batch_queue.put(None)
        executor.shutdown(wait=True)
        upload_pbar.close()
        return 0

    # Send poison pills to stop workers
    for _ in range(num_threads):
        batch_queue.put(None)

    # Wait for all uploads to complete
    executor.shutdown(wait=True)
    upload_pbar.close()
    uploaded = sum(results)

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("UPLOAD COMPLETE")
    print("=" * 60)
    print(f"✓ Uploaded: {uploaded:,} points")
    print(f"✓ Collection: {collection_name}")
    print(f"✓ Time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"✓ Speed: {uploaded / elapsed:.0f} points/sec")

    # Verify
    try:
        collection_info = client.get_collection(collection_name)
        print(f"\n✓ Qdrant reports {collection_info.points_count:,} points in collection")
    except Exception as e:
        print(f"\n⚠️  Could not verify collection: {e}")

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