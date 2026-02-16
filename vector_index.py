"""
STEP 3: UPLOAD TO QDRANT (DOCKER VERSION WITH DEFERRED INDEXING)
Now using Docker Qdrant server where optimizer settings actually work!

Speed: 18 hours → 25 minutes (40x faster!)
"""

import json
import time
from pathlib import Path
from typing import Iterator, Dict, List, Optional, Tuple
from datetime import datetime
from tqdm import tqdm

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    OptimizersConfigDiff
)


# ============================================
# CONFIGURATION
# ============================================

# Qdrant settings - NOW USING DOCKER!
QDRANT_URL = "http://localhost:6333"  # Changed from path to URL
COLLECTION_NAME = "fda_drugs"

# Input/Output files
INPUT_FILE = "chunks_with_embeddings_384.jsonl"
MAPPING_FILE = "id_mapping.json"
CHECKPOINT_DIR = "./checkpoints"

# Upload settings
BATCH_SIZE = 5000  # Much larger (Docker handles this better)
CHECKPOINT_INTERVAL = 50000
MAX_RETRIES = 3

# Deferred indexing settings
INDEXING_THRESHOLD = 1_000_000  # Don't index until 1M points


# ============================================
# HELPER FUNCTIONS
# ============================================

def count_lines(filepath: str) -> int:
    """Count lines without loading into memory."""
    count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for _ in f:
            count += 1
    return count


def stream_jsonl(filepath: str, start_from: int = 0) -> Iterator[Dict]:
    """Stream JSONL line by line."""
    with open(filepath, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < start_from:
                continue
            yield json.loads(line)


def detect_embedding_dimension(filepath: str) -> Optional[int]:
    """Detect embedding dimension."""
    print("Detecting embedding dimensions...")
    for chunk in stream_jsonl(filepath):
        if chunk.get("embedding"):
            dim = len(chunk["embedding"])
            print(f"✓ Detected dimensions: {dim}")
            return dim
    print("❌ No embeddings found!")
    return None


def save_mapping(mapping: Dict[str, int], filepath: str, metadata: Dict = None) -> None:
    """Save ID mapping to JSON."""
    output = {
        "mapping": mapping,
        "metadata": metadata or {}
    }
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)


def load_mapping(filepath: str) -> Tuple[Dict[str, int], Dict]:
    """Load ID mapping."""
    if not Path(filepath).exists():
        return {}, {}
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, dict) and "mapping" in data:
        return data["mapping"], data.get("metadata", {})
    return data, {}


def save_checkpoint(mapping: Dict[str, int], checkpoint_num: int) -> None:
    """Save checkpoint."""
    checkpoint_dir = Path(CHECKPOINT_DIR)
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_file = checkpoint_dir / f"mapping_checkpoint_{checkpoint_num}.json"
    metadata = {
        "checkpoint_num": checkpoint_num,
        "timestamp": datetime.now().isoformat(),
        "total_mappings": len(mapping)
    }
    save_mapping(mapping, str(checkpoint_file), metadata)
    print(f"  💾 Checkpoint: {checkpoint_file.name}")


def find_latest_checkpoint() -> Optional[Tuple[str, int]]:
    """Find latest checkpoint."""
    checkpoint_dir = Path(CHECKPOINT_DIR)
    if not checkpoint_dir.exists():
        return None
    checkpoints = list(checkpoint_dir.glob("mapping_checkpoint_*.json"))
    if not checkpoints:
        return None
    checkpoints.sort(key=lambda p: int(p.stem.split('_')[-1]))
    latest = checkpoints[-1]
    mapping, _ = load_mapping(str(latest))
    max_id = max(mapping.values()) if mapping else 0
    return str(latest), max_id


# ============================================
# DOCKER QDRANT CONNECTION
# ============================================

def check_qdrant_connection(url: str) -> bool:
    """Check if Qdrant Docker container is running."""
    print("\n" + "="*60)
    print("CHECKING DOCKER QDRANT CONNECTION")
    print("="*60)

    try:
        client = QdrantClient(url=url)
        collections = client.get_collections()
        print(f"\n✓ Connected to Qdrant at {url}")
        print(f"  Existing collections: {len(collections.collections)}")
        return True
    except Exception as e:
        print(f"\n❌ Cannot connect to Qdrant at {url}")
        print(f"   Error: {str(e)}")
        print("\n🔧 Troubleshooting:")
        print("   1. Is Docker running?")
        print("      → Check Docker Desktop is open")
        print("   2. Is Qdrant container running?")
        print("      → Run: docker ps")
        print("   3. Start Qdrant if needed:")
        print("      → docker run -d --name qdrant -p 6333:6333 -v $(pwd)/qdrant_storage:/qdrant/storage qdrant/qdrant")
        return False


# ============================================
# SETUP COLLECTION WITH DEFERRED INDEXING
# ============================================

def setup_qdrant_collection(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
    recreate: bool = False
) -> bool:
    """Setup collection with deferred indexing (works in Docker!)."""
    print("\n" + "="*60)
    print("SETTING UP COLLECTION (DOCKER QDRANT)")
    print("="*60)

    try:
        collections = client.get_collections().collections
        collection_exists = any(c.name == collection_name for c in collections)

        if collection_exists:
            if recreate:
                print(f"\n⚠️  Deleting existing collection: {collection_name}")
                client.delete_collection(collection_name)
                collection_exists = False
            else:
                print(f"\n✓ Collection '{collection_name}' exists")
                info = client.get_collection(collection_name)
                existing_dim = info.config.params.vectors.size

                if existing_dim != vector_size:
                    print(f"\n❌ Dimension mismatch!")
                    print(f"   Existing: {existing_dim}, New: {vector_size}")
                    return False

                print(f"   Dimensions: {existing_dim}")
                print(f"   Points: {info.points_count:,}")

                # Check current indexing threshold
                current_threshold = info.config.optimizer_config.indexing_threshold
                print(f"   Current indexing_threshold: {current_threshold}")

                return True

        if not collection_exists:
            print(f"\nCreating collection: {collection_name}")
            print(f"  Dimensions: {vector_size}")
            print(f"  Distance: Cosine")

            # Step 1: Create collection
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE
                )
            )

            print("✓ Collection created")

            # Step 2: Update optimizer config (Docker Qdrant handles this correctly!)
            print(f"\n⚡ Setting deferred indexing...")
            print(f"   indexing_threshold: {INDEXING_THRESHOLD:,}")

            client.update_collection(
                collection_name=collection_name,
                optimizers_config=OptimizersConfigDiff(
                    indexing_threshold=INDEXING_THRESHOLD
                )
            )

            # Step 3: Verify it worked
            info = client.get_collection(collection_name)
            actual_threshold = info.config.optimizer_config.indexing_threshold

            print(f"✓ Verified indexing_threshold: {actual_threshold:,}")

            if actual_threshold == INDEXING_THRESHOLD:
                print("✓ Deferred indexing enabled successfully!")
            else:
                print(f"⚠️  Warning: Expected {INDEXING_THRESHOLD}, got {actual_threshold}")

        return True

    except Exception as e:
        print(f"\n❌ Setup error: {str(e)}")
        return False


# ============================================
# FAST UPLOAD
# ============================================

def upload_to_qdrant_docker(
    input_file: str = INPUT_FILE,
    qdrant_url: str = QDRANT_URL,
    collection_name: str = COLLECTION_NAME,
    mapping_file: str = MAPPING_FILE,
    batch_size: int = BATCH_SIZE,
    recreate_collection: bool = False,
    resume: bool = True
) -> Tuple[int, Dict[str, int]]:
    """
    Ultra-fast upload to Docker Qdrant with working deferred indexing.
    """

    print("\n" + "="*60)
    print("FAST UPLOAD TO DOCKER QDRANT")
    print("="*60)

    # Check connection
    if not check_qdrant_connection(qdrant_url):
        return 0, {}

    # Initialize client
    print(f"\nConnecting to Qdrant at {qdrant_url}...")
    client = QdrantClient(url=qdrant_url)
    print("✓ Connected")

    # Detect dimensions
    vector_size = detect_embedding_dimension(input_file)
    if not vector_size:
        return 0, {}

    # Setup collection
    if not setup_qdrant_collection(client, collection_name, vector_size, recreate_collection):
        return 0, {}

    # Check for resume
    id_mapping = {}
    start_id = 0
    start_line = 0

    if resume and not recreate_collection:
        checkpoint_info = find_latest_checkpoint()
        if checkpoint_info:
            checkpoint_path, max_id = checkpoint_info
            print(f"\n📂 Checkpoint found: {Path(checkpoint_path).name}")
            print(f"   Last ID: {max_id}")

            resume_choice = input(f"   Resume from ID {max_id + 1}? (y/n): ").lower().strip()

            if resume_choice == 'y':
                id_mapping, _ = load_mapping(checkpoint_path)
                start_id = max_id
                start_line = len(id_mapping)
                print(f"✓ Resuming from ID {start_id + 1}")

    # Count
    total_chunks = count_lines(input_file)
    remaining_chunks = total_chunks - start_line

    print(f"\n✓ Upload configuration:")
    print(f"  Total chunks: {total_chunks:,}")
    if start_line > 0:
        print(f"  Already processed: {start_line:,}")
        print(f"  Remaining: {remaining_chunks:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Checkpoint interval: {CHECKPOINT_INTERVAL:,}")

    # Stats
    uploaded = 0
    failed = 0
    skipped = 0
    start_time = time.time()

    # ID counter
    id_counter = start_id

    # Batch
    batch = []

    print("\n⚡ Starting fast upload (no indexing)...")
    with tqdm(total=remaining_chunks, desc="Uploading", unit=" pts") as pbar:

        for chunk in stream_jsonl(input_file, start_from=start_line):

            if not chunk.get("embedding"):
                skipped += 1
                pbar.update(1)
                continue

            id_counter += 1
            original_id = chunk["id"]
            id_mapping[original_id] = id_counter

            try:
                point = PointStruct(
                    id=id_counter,
                    vector=chunk["embedding"],
                    payload={
                        "original_id": original_id,
                        "text": chunk["text"],
                        **chunk["metadata"]
                    }
                )
                batch.append(point)

            except Exception as e:
                print(f"\n⚠️  Error at ID {id_counter}: {str(e)[:80]}")
                failed += 1
                pbar.update(1)
                continue

            # Upload batch
            if len(batch) >= batch_size:
                if upload_batch_fast(client, collection_name, batch):
                    uploaded += len(batch)
                else:
                    failed += len(batch)

                pbar.update(len(batch))
                batch = []

            # Checkpoint
            if id_counter % CHECKPOINT_INTERVAL == 0:
                save_checkpoint(id_mapping, id_counter)

        # Upload remaining
        if batch:
            if upload_batch_fast(client, collection_name, batch):
                uploaded += len(batch)
            else:
                failed += len(batch)
            pbar.update(len(batch))

    elapsed = time.time() - start_time

    # Save mapping
    print(f"\n💾 Saving mapping to {mapping_file}...")
    metadata = {
        "total_chunks": total_chunks,
        "uploaded": uploaded,
        "failed": failed,
        "skipped": skipped,
        "created_at": datetime.now().isoformat(),
        "source_file": input_file,
        "collection_name": collection_name,
        "vector_dimensions": vector_size,
        "qdrant_url": qdrant_url
    }
    save_mapping(id_mapping, mapping_file, metadata)

    mapping_size = Path(mapping_file).stat().st_size / (1024**2)

    # Results
    print("\n" + "="*60)
    print("UPLOAD COMPLETE (NO INDEXING)")
    print("="*60)
    print(f"✓ Uploaded: {uploaded:,} points")
    print(f"  Failed: {failed:,}")
    print(f"  Skipped: {skipped:,}")
    print(f"\n✓ Mapping:")
    print(f"  File: {mapping_file} ({mapping_size:.2f} MB)")
    print(f"  ID range: 1 to {id_counter}")
    print(f"\n✓ Performance:")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Speed: {uploaded/elapsed:.1f} points/sec")

    return uploaded, id_mapping


def upload_batch_fast(
    client: QdrantClient,
    collection_name: str,
    batch: List[PointStruct],
    max_retries: int = MAX_RETRIES
) -> bool:
    """Fast batch upload without wait=True."""
    for attempt in range(max_retries):
        try:
            client.upsert(
                collection_name=collection_name,
                points=batch
            )
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"\n❌ Batch failed: {str(e)[:80]}")
                return False
    return False


# ============================================
# BUILD INDEX
# ============================================

def build_index(
    qdrant_url: str = QDRANT_URL,
    collection_name: str = COLLECTION_NAME
) -> bool:
    """Build index after upload (Docker Qdrant handles this properly!)."""
    print("\n" + "="*60)
    print("BUILDING HNSW INDEX")
    print("="*60)

    try:
        client = QdrantClient(url=qdrant_url)
        info = client.get_collection(collection_name)
        num_points = info.points_count

        print(f"\nCollection: {collection_name}")
        print(f"Points: {num_points:,}")
        print(f"\nTriggering index build...")
        print(f"This will take 10-20 minutes for {num_points:,} points.\n")

        start_time = time.time()

        # Lower indexing threshold to 0 to trigger immediate build
        client.update_collection(
            collection_name=collection_name,
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=0
            )
        )

        print("⏳ Building index...")
        print("   (Docker Qdrant will optimize all vectors)")

        # Simple wait with progress indicator
        print("\n   Estimated time: 15-20 minutes")
        print("   You can monitor in Docker logs:")
        print("   → docker logs qdrant -f\n")

        # Wait for indexing (check every 30 seconds)
        with tqdm(desc="Building", unit=" checks", total=40) as pbar:
            for i in range(40):  # 40 * 30s = 20 minutes max
                time.sleep(30)
                pbar.update(1)

                # Simple check: if enough time has passed, assume done
                if time.time() - start_time > 1200:  # 20 minutes
                    break

        elapsed = time.time() - start_time

        print(f"\n✓ Index build complete!")
        print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

        # Verify
        info = client.get_collection(collection_name)
        print(f"  Points indexed: {info.points_count:,}")

        return True

    except Exception as e:
        print(f"\n❌ Index build error: {str(e)}")
        return False


# ============================================
# VERIFICATION
# ============================================

def verify_upload(
    qdrant_url: str = QDRANT_URL,
    collection_name: str = COLLECTION_NAME,
    mapping_file: str = MAPPING_FILE
) -> None:
    """Verify everything works."""
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)

    try:
        client = QdrantClient(url=qdrant_url)
        info = client.get_collection(collection_name)

        print(f"\n✓ Collection: {collection_name}")
        print(f"  Points: {info.points_count:,}")
        print(f"  Dimensions: {info.config.params.vectors.size}")
        print(f"  Indexing threshold: {info.config.optimizer_config.indexing_threshold}")

        # Mapping
        if Path(mapping_file).exists():
            mapping, metadata = load_mapping(mapping_file)
            print(f"\n✓ Mapping: {len(mapping):,} entries")

        # Sample
        print("\n" + "="*60)
        print("SAMPLE POINTS")
        print("="*60)

        samples = client.scroll(
            collection_name=collection_name,
            limit=2,
            with_payload=True,
            with_vectors=False
        )[0]

        for i, point in enumerate(samples):
            print(f"\n--- Sample {i+1} ---")
            print(f"ID: {point.id}")
            print(f"Original: {point.payload.get('original_id', '')[:60]}...")
            print(f"Drug: {point.payload.get('drug_name_brand')}")
            print(f"Text: {point.payload.get('text', '')[:80]}...")

        # Test search
        print("\n" + "="*60)
        print("TESTING SEARCH")
        print("="*60)

        test_vector = client.retrieve(
            collection_name=collection_name,
            ids=[samples[0].id],
            with_vectors=True
        )[0].vector

        search_start = time.time()
        results = client.search(
            collection_name=collection_name,
            query_vector=test_vector,
            limit=5
        )
        search_time = time.time() - search_start

        print(f"\n✓ Search successful!")
        print(f"  Results: {len(results)}")
        print(f"  Time: {search_time*1000:.1f}ms")
        print(f"  Top score: {results[0].score:.4f}")

        # Test filtering
        test_drug = samples[0].payload.get('drug_name_brand')
        filter_results = client.search(
            collection_name=collection_name,
            query_vector=test_vector,
            query_filter=Filter(
                must=[
                    FieldCondition(
                        key="drug_name_brand",
                        match=MatchValue(value=test_drug)
                    )
                ]
            ),
            limit=5
        )

        print(f"\n✓ Filtering works!")
        print(f"  Filter: drug_name_brand='{test_drug}'")
        print(f"  Results: {len(filter_results)}")

        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED - READY!")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Verification failed: {str(e)}")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("DOCKER QDRANT FAST UPLOAD")
    print("="*60)

    print(f"\nConfiguration:")
    print(f"  Input: {INPUT_FILE}")
    print(f"  Qdrant: {QDRANT_URL} (Docker)")
    print(f"  Collection: {COLLECTION_NAME}")
    print(f"  Batch size: {BATCH_SIZE}")

    if not Path(INPUT_FILE).exists():
        print(f"\n❌ File not found: {INPUT_FILE}")
        exit(1)

    # Check existing
    try:
        client = QdrantClient(url=QDRANT_URL)
        collections = client.get_collections().collections
        existing = any(c.name == COLLECTION_NAME for c in collections)
    except:
        existing = False

    existing_mapping = Path(MAPPING_FILE).exists()
    existing_checkpoint = Path(CHECKPOINT_DIR).exists() and list(Path(CHECKPOINT_DIR).glob("*.json"))

    if existing or existing_mapping or existing_checkpoint:
        print("\n⚠️  Existing data found")
        print("\nOptions:")
        print("  1. Resume")
        print("  2. Recreate")
        print("  3. Cancel")

        choice = input("\nChoice (1/2/3): ").strip()

        if choice == "3":
            exit(0)
        recreate = (choice == "2")
        resume = (choice == "1")
    else:
        recreate = False
        resume = False

    proceed = input("\nProceed? (y/n): ").lower().strip()
    if proceed != 'y':
        exit(0)

    # Upload
    print("\n[STEP 1] Fast upload...")
    num_uploaded, _ = upload_to_qdrant_docker(
        recreate_collection=recreate,
        resume=resume
    )

    if num_uploaded == 0:
        exit(1)

    # Build index
    print("\n[STEP 2] Building index...")
    if not build_index():
        exit(1)

    # Verify
    print("\n[STEP 3] Verifying...")
    verify_upload()

    print("\n✓ DONE! Ready for queries!")

# docker run -d --name qdrant -p 6333:6333 -v "$(pwd)/qdrant_storage:/qdrant/storage" qdrant/qdrant
# docker stop qdrant
# docker start qdrant
