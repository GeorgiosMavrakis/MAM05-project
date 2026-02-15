"""
STEP 3: UPLOAD EMBEDDINGS TO QDRANT
Uploads chunks with embeddings to local Qdrant database.

Features:
- Automatic dimension detection
- Streaming upload (memory efficient)
- Progress tracking
- Error handling and retry logic
- Collection management
- Verification after upload
"""

import json
import time
from pathlib import Path
from typing import Iterator, Dict, List, Optional
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue
)


# ============================================
# CONFIGURATION
# ============================================

# Qdrant settings
QDRANT_PATH = "./qdrant_data"  # Local storage path
COLLECTION_NAME = "rag_collection"  # "fda_drugs"

# Input file
INPUT_FILE = "chunks_with_embeddings_384.jsonl"

# Upload settings
BATCH_SIZE = 100  # Upload 100 points at a time
MAX_RETRIES = 3  # Retry failed batches


# ============================================
# HELPER FUNCTIONS
# ============================================

def count_lines(filepath: str) -> int:
    """Count lines in file without loading into memory."""
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for _ in f:
            count += 1
    return count


def stream_jsonl(filepath: str) -> Iterator[Dict]:
    """Stream JSONL file line by line (memory efficient)."""
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)


def detect_embedding_dimension(filepath: str) -> Optional[int]:
    """
    Detect embedding dimension by reading first chunk with embedding.

    Returns:
        Dimension size (e.g., 384, 1536) or None if no embeddings found
    """
    print("Detecting embedding dimensions...")

    for chunk in stream_jsonl(filepath):
        if chunk.get("embedding"):
            dim = len(chunk["embedding"])
            print(f"✓ Detected dimensions: {dim}")
            return dim

    print("❌ No embeddings found in file!")
    return None


# ============================================
# QDRANT SETUP
# ============================================

def setup_qdrant_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    recreate: bool = False
) -> bool:
    """
    Create or verify Qdrant collection.

    Args:
        client: Qdrant client
        collection_name: Name of collection
        vector_size: Embedding dimensions (384, 1536, etc.)
        recreate: If True, delete existing collection and recreate

    Returns:
        True if successful
    """
    print("\n" + "="*60)
    print("SETTING UP QDRANT COLLECTION")
    print("="*60)

    try:
        # Check if collection exists
        collections = client.get_collections().collections
        collection_exists = any(c.name == collection_name for c in collections)

        if collection_exists:
            if recreate:
                print(f"\n⚠️  Deleting existing collection: {collection_name}")
                client.delete_collection(collection_name)
                collection_exists = False
            else:
                print(f"\n✓ Collection '{collection_name}' already exists")

                # Verify vector dimensions match
                collection_info = client.get_collection(collection_name)
                existing_dim = collection_info.config.params.vectors.size

                if existing_dim != vector_size:
                    print(f"\n❌ ERROR: Dimension mismatch!")
                    print(f"   Existing collection: {existing_dim} dims")
                    print(f"   New embeddings: {vector_size} dims")
                    print(f"\n   Options:")
                    print(f"   1. Recreate collection (set recreate=True)")
                    print(f"   2. Use different collection name")
                    return False

                print(f"   Dimensions: {existing_dim}")
                return True

        if not collection_exists:
            print(f"\nCreating new collection: {collection_name}")
            print(f"  Vector dimensions: {vector_size}")
            print(f"  Distance metric: Cosine")

            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE  # Best for normalized embeddings
                )
            )

            print(f"✓ Collection created successfully")

        return True

    except Exception as e:
        print(f"\n❌ Error setting up collection: {str(e)}")
        return False


# ============================================
# UPLOAD TO QDRANT
# ============================================

def upload_to_qdrant(
    input_file: str = INPUT_FILE,
    qdrant_path: str = QDRANT_PATH,
    collection_name: str = COLLECTION_NAME,
    batch_size: int = BATCH_SIZE,
    recreate_collection: bool = False
) -> int:
    """
    Upload embeddings to Qdrant with streaming (memory efficient).

    Args:
        input_file: Path to chunks_with_embeddings.jsonl
        qdrant_path: Path to Qdrant storage directory
        collection_name: Name of Qdrant collection
        batch_size: Number of points to upload per batch
        recreate_collection: If True, recreate collection from scratch

    Returns:
        Number of points uploaded
    """

    print("\n" + "="*60)
    print("UPLOADING TO QDRANT")
    print("="*60)

    # Initialize Qdrant client
    print(f"\nInitializing Qdrant client...")
    print(f"  Storage path: {qdrant_path}")

    client = QdrantClient(path=qdrant_path)
    print("✓ Client initialized")

    # Detect embedding dimensions
    vector_size = detect_embedding_dimension(input_file)
    if not vector_size:
        print("\n❌ Cannot proceed without embeddings")
        return 0

    # Setup collection
    if not setup_qdrant_collection(
        client, collection_name, vector_size, recreate_collection
    ):
        return 0

    # Count total chunks
    total_chunks = count_lines(input_file)
    print(f"\n✓ Total chunks to upload: {total_chunks:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Estimated batches: {(total_chunks + batch_size - 1) // batch_size}")

    # Upload statistics
    uploaded = 0
    failed = 0
    skipped = 0  # Chunks without embeddings
    start_time = time.time()

    # Process in batches
    batch = []

    print("\nUploading...")
    with tqdm(total=total_chunks, desc="Uploading") as pbar:

        for chunk in stream_jsonl(input_file):

            # Skip chunks without embeddings
            if not chunk.get("embedding"):
                skipped += 1
                pbar.update(1)
                continue

            # Create Qdrant point
            try:
                point = PointStruct(
                    id=chunk["id"],  # Use chunk ID as point ID
                    vector=chunk["embedding"],
                    payload={
                        # Store text (for LLM context)
                        "text": chunk["text"],

                        # Store ALL metadata (for filtering)
                        **chunk["metadata"]
                    }
                )
                batch.append(point)

            except Exception as e:
                print(f"\n⚠️  Error creating point for {chunk.get('id')}: {str(e)}")
                failed += 1
                pbar.update(1)
                continue

            # Upload batch when full
            if len(batch) >= batch_size:
                success = upload_batch(client, collection_name, batch)

                if success:
                    uploaded += len(batch)
                else:
                    failed += len(batch)

                pbar.update(len(batch))
                batch = []

        # Upload remaining batch
        if batch:
            success = upload_batch(client, collection_name, batch)

            if success:
                uploaded += len(batch)
            else:
                failed += len(batch)

            pbar.update(len(batch))

    elapsed = time.time() - start_time

    # Results
    print("\n" + "="*60)
    print("UPLOAD COMPLETE")
    print("="*60)
    print(f"✓ Successfully uploaded: {uploaded:,}")
    print(f"  Failed: {failed:,}")
    print(f"  Skipped (no embedding): {skipped:,}")
    print(f"\n✓ Collection: {collection_name}")
    print(f"  Storage: {qdrant_path}")
    print(f"\n✓ Performance:")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Speed: {uploaded/elapsed:.1f} points/sec")

    # Get storage size
    qdrant_size = sum(
        f.stat().st_size for f in Path(qdrant_path).rglob('*') if f.is_file()
    ) / (1024**2)  # MB
    print(f"\n✓ Qdrant storage size: {qdrant_size:.1f} MB")

    return uploaded


def upload_batch(
    client: QdrantClient,
    collection_name: str,
    batch: List[PointStruct],
    max_retries: int = MAX_RETRIES
) -> bool:
    """
    Upload a batch of points with retry logic.

    Args:
        client: Qdrant client
        collection_name: Collection to upload to
        batch: List of points
        max_retries: Maximum retry attempts

    Returns:
        True if successful
    """
    for attempt in range(max_retries):
        try:
            client.upsert(
                collection_name=collection_name,
                points=batch
            )
            return True

        except Exception as e:
            if attempt < max_retries - 1:
                print(f"\n⚠️  Batch upload failed (attempt {attempt + 1}), retrying...")
                time.sleep(1)  # Wait before retry
            else:
                print(f"\n❌ Batch upload failed after {max_retries} attempts: {str(e)}")
                return False

    return False


# ============================================
# VERIFICATION
# ============================================

def verify_qdrant_upload(
    qdrant_path: str = QDRANT_PATH,
    collection_name: str = COLLECTION_NAME
) -> None:
    """
    Verify the uploaded data in Qdrant.
    """
    print("\n" + "="*60)
    print("VERIFYING QDRANT COLLECTION")
    print("="*60)

    try:
        client = QdrantClient(path=qdrant_path)

        # Get collection info
        collection_info = client.get_collection(collection_name)

        print(f"\n✓ Collection: {collection_name}")
        print(f"  Points count: {collection_info.points_count:,}")
        print(f"  Vector dimensions: {collection_info.config.params.vectors.size}")
        print(f"  Distance metric: {collection_info.config.params.vectors.distance}")

        # Get a sample point
        print("\n" + "="*60)
        print("SAMPLE POINTS")
        print("="*60)

        # Scroll through first 3 points
        sample_points = client.scroll(
            collection_name=collection_name,
            limit=3,
            with_payload=True,
            with_vectors=False  # Don't return vectors (too large to display)
        )[0]

        for i, point in enumerate(sample_points):
            print(f"\n--- Sample {i+1} ---")
            print(f"ID: {point.id}")
            print(f"Drug: {point.payload.get('drug_name_brand')}")
            print(f"Category: {point.payload.get('category')}")
            print(f"Text: {point.payload.get('text', '')[:100]}...")

            # Show all metadata keys
            metadata_keys = [k for k in point.payload.keys() if k != 'text']
            print(f"Metadata keys: {', '.join(metadata_keys)}")

        # Test search functionality
        print("\n" + "="*60)
        print("TESTING SEARCH FUNCTIONALITY")
        print("="*60)

        # Get first point's vector for testing
        test_point = client.retrieve(
            collection_name=collection_name,
            ids=[sample_points[0].id],
            with_vectors=True
        )[0]

        # Search with this vector
        search_results = client.search(
            collection_name=collection_name,
            query_vector=test_point.vector,
            limit=3
        )

        print(f"\n✓ Search test successful!")
        print(f"  Query: First point's vector")
        print(f"  Results: {len(search_results)}")
        print(f"  Top match score: {search_results[0].score:.4f}")

        # Test metadata filtering
        print("\n" + "="*60)
        print("TESTING METADATA FILTERING")
        print("="*60)

        # Try filtering by drug name
        test_drug = sample_points[0].payload.get('drug_name_brand')

        filter_results = client.search(
            collection_name=collection_name,
            query_vector=test_point.vector,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="drug_name_brand",
                        match=MatchValue(value=test_drug)
                    )
                ]
            ),
            limit=3
        )

        print(f"\n✓ Filtering test successful!")
        print(f"  Filter: drug_name_brand = '{test_drug}'")
        print(f"  Results: {len(filter_results)}")

        print("\n" + "="*60)
        print("✓ VERIFICATION COMPLETE - ALL TESTS PASSED!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Verification failed: {str(e)}")


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("QDRANT UPLOAD SCRIPT")
    print("="*60)

    # Configuration check
    print("\nConfiguration:")
    print(f"  Input file: {INPUT_FILE}")
    print(f"  Qdrant path: {QDRANT_PATH}")
    print(f"  Collection name: {COLLECTION_NAME}")

    if not Path(INPUT_FILE).exists():
        print(f"\n❌ Input file not found: {INPUT_FILE}")
        exit(1)

    # Ask about recreation
    print("\n" + "="*60)
    print("Would you like to recreate the collection?")
    print("  - Yes: Delete existing data and start fresh")
    print("  - No: Add to existing collection (or create if doesn't exist)")
    recreate = input("Recreate? (y/n): ").lower().strip() == 'y'

    # Confirm
    print("\n" + "="*60)
    proceed = input("\nProceed with upload? (y/n): ").lower().strip()

    if proceed != 'y':
        print("Cancelled.")
        exit(0)

    # Upload
    print("\n[STEP 1] Uploading to Qdrant...")
    num_uploaded = upload_to_qdrant(recreate_collection=recreate)

    if num_uploaded == 0:
        print("\n❌ Upload failed or no data uploaded")
        exit(1)

    # Verify
    print("\n[STEP 2] Verifying upload...")
    verify_qdrant_upload()

    print("\n" + "="*60)
    print("✓ ALL DONE!")
    print("="*60)
    print(f"✓ Uploaded {num_uploaded:,} points to Qdrant")
    print(f"✓ Ready for queries!")
    print("\nNext: Implement query interface (STEP 4)")
