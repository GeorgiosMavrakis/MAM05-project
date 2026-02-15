"""
Produce numeric embeddings for texts

Purpose: convert chunk texts (and query strings) into numeric vectors.

Input: list of strings ["text1", "text2", ..."]

Output: list of vectors [[0.12, ...], [0.23, ...]] (floats)

Example: embed_texts(["dizziness reported in trials"]) → [[0.013, -0.22, ...]]

Notes: batch embedding offline for entire corpus; use same model for query embedding.
"""
# from typing import List
# from sentence_transformers import SentenceTransformer
# import config, chunker, database
# import os
#
# embedder = SentenceTransformer("all-MiniLM-L6-v2")
#
#
# def get_embedding(text: str) -> List[float]:
#     vector = embedding().encode(text)
#
#     return vector.tolist()
#
# def embedding():
#
#     with open(DATA_FILE, "r", encoding="utf-8") as f: # TODO Add file path
#         text = f.read()
#
#     chunks = chunker.chunk_text(text)
#
#     ids = []
#     docs = []
#     embeds = []
#
#     for i, chunk in enumerate(chunks):
#
#         ids.append(str(i))
#         docs.append(chunk)
#         embeds.append(get_embedding(chunk))
#
#     database.collection.add(
#         ids=ids,
#         documents=docs,
#         embeddings=embeds
#     )
#     database.chroma_client.persist()

"""
STEP 2: GENERATE EMBEDDINGS USING OPENAI (UvA Azure Proxy)
Adapted for UvA's Azure OpenAI proxy endpoint.

Features:
- Uses UvA's custom Azure proxy
- Checks available embedding models
- Generates embeddings with progress tracking
- Saves embeddings to JSONL
"""

import json, gc
import time
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
from openai import OpenAI


# ============================================
# CONFIGURATION - UvA AZURE PROXY
# ============================================

# UvA Azure Proxy Configuration
UVA_API_ENDPOINT = "https://ai-research-proxy.azurewebsites.net"
UVA_API_KEY = "sk-zRBvmHTTr1dZyrQg2_6mlg"  # Your UvA API key

# Embedding model - Check what's available on UvA proxy
# Common Azure OpenAI embedding models:
# - "text-embedding-ada-002" (most common)
# - "text-embedding-3-small" (if available)
# - "text-embedding-3-large" (if available)
EMBEDDING_MODEL = "text-embedding-ada-002"  # Start with this, adjust if needed

# File paths
INPUT_FILE = "chunks.jsonl"
OUTPUT_FILE = "chunks_with_embeddings.jsonl"

# Batch processing settings
BATCH_SIZE = 300  # Process 100 chunks per API call
RATE_LIMIT_DELAY = 4.5  # 500ms delay (Azure can be more restrictive)


# ============================================
# HELPER FUNCTIONS
# ============================================

def load_jsonl(filepath: str) -> List[Dict]:
    """Load JSONL file into a list (loads entire file into memory)."""
    chunks: List[Dict] = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def save_jsonl(data: List[Dict], filepath: str) -> None:
    """Save data to JSONL format."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def count_tokens_estimate(text: str) -> int:
    """Rough token count estimate."""
    return int(len(text.split()) * 1.3)


# ============================================
# UVA AZURE PROXY - MODEL CHECKING
# ============================================

import time
import gc
import json
from pathlib import Path
from tqdm import tqdm

def check_available_embedding_models(api_key: str, base_url: str) -> bool:
    """
    Check which embedding models are available on the UvA Azure proxy.
    Returns True if at least one common embedding model works, False otherwise.
    """
    print("\n" + "="*60)
    print("CHECKING UVA AZURE OPENAI API ACCESS")
    print("="*60)

    try:
        # Initialize client with UvA proxy settings
        client = OpenAI(
            api_key=api_key,
            base_url=base_url
        )

        print("\nAttempting to list available models...")

        try:
            # Try to list models (might not be supported by proxy)
            models = client.models.list()

            embedding_models = []
            for model in models.data:
                if "embedding" in model.id.lower() or "embed" in model.id.lower():
                    embedding_models.append(model)

            if embedding_models:
                print(f"\n✓ Found {len(embedding_models)} embedding model(s):")
                print("\n" + "-"*60)
                for model in embedding_models:
                    print(f"  Model ID: {model.id}")
                    if hasattr(model, 'created') and model.created:
                        try:
                            print(f"  Created: {time.strftime('%Y-%m-%d', time.gmtime(model.created))}")
                        except Exception:
                            pass
                    print("-"*60)
            else:
                print("\n⚠️  Model listing not supported or no embedding models found.")
                print("Will test common embedding models...")

        except Exception as list_error:
            print(f"\n⚠️  Model listing not supported by proxy: {str(list_error)}")
            print("This is normal for Azure proxies. Will test common models...")

        # Test common embedding models
        print("\n" + "="*60)
        print("TESTING COMMON EMBEDDING MODELS")
        print("="*60)

        test_models = [
            "text-embedding-ada-002",
            "text-embedding-3-small",
            "text-embedding-3-large"
        ]

        working_models = []

        for model_name in test_models:
            print(f"\nTesting: {model_name}...", end=" ")
            try:
                response = client.embeddings.create(
                    model=model_name,
                    input="test"
                )
                embedding_dims = len(response.data[0].embedding)
                print(f"✓ WORKS! ({embedding_dims} dimensions)")
                working_models.append((model_name, embedding_dims))
            except Exception as e:
                error_msg = str(e)
                if "model" in error_msg.lower() or "not found" in error_msg.lower():
                    print(f"✗ Not available")
                else:
                    print(f"✗ Error: {error_msg[:50]}...")

        if not working_models:
            print("\n❌ No embedding models are working!")
            print("\nPlease contact your supervisor to confirm:")
            print("  1. Your API key has access to embedding models")
            print("  2. Which embedding models are available on the UvA proxy")
            return False

        # Show working models
        print("\n" + "="*60)
        print("AVAILABLE EMBEDDING MODELS")
        print("="*60)
        for model_name, dims in working_models:
            print(f"  ✓ {model_name} ({dims} dimensions)")
            if model_name == "text-embedding-3-small":
                print(f"    → RECOMMENDED: Best balance")
            elif model_name == "text-embedding-ada-002":
                print(f"    → RELIABLE: Most compatible")

        # Cost estimate
        print("\n" + "="*60)
        print("COST ESTIMATE FOR YOUR DATASET")
        print("="*60)

        if Path(INPUT_FILE).exists():
            # stream-count chunks to avoid loading them fully
            total_chunks = 0
            with open(INPUT_FILE, "r", encoding="utf-8") as f:
                for _ in f:
                    total_chunks += 1

            # Estimate tokens conservatively by sampling if count_tokens_estimate is expensive.
            # Here we compute tokens for all chunks using your existing function (keeps memory low).
            total_tokens = 0
            with open(INPUT_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                        total_tokens += count_tokens_estimate(chunk.get("text", ""))
                    except Exception:
                        # skip malformed lines for the estimate
                        continue

            num_batches = (total_chunks + BATCH_SIZE - 1) // BATCH_SIZE
            estimated_minutes = num_batches * RATE_LIMIT_DELAY / 60.0

            print(f"  Total chunks: {total_chunks:,}")
            print(f"  Estimated tokens: {total_tokens:,}")
            print(f"  Estimated batches: {num_batches:,}")
            print(f"  Processing time (rate-limit waits only): ~{estimated_minutes:.1f} minutes")
            print(f"\n  ✓ Cost: FREE (provided by UvA)")

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        print("\nPossible issues:")
        print("  1. Invalid UvA API key")
        print("  2. Network connectivity problem")
        print("  3. UvA proxy service issue")
        print("  4. API key not activated")
        print("\nPlease verify:")
        print("  - Your UvA API key is correct")
        print("  - You're connected to the internet (or UvA VPN if required)")
        print("  - Contact your supervisor if issues persist")

        return False


def stream_jsonl_batches(input_file: str, batch_size: int):
    """
    Generator that yields batches of dicts from a JSONL file without loading the whole file into memory.
    """
    batch = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            batch.append(json.loads(line))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


# ============================================
# MAIN EMBEDDING GENERATION
# ============================================

def generate_embeddings_openai_uva(
    input_file: str = INPUT_FILE,
    output_file: str = OUTPUT_FILE,
    api_key: str = UVA_API_KEY,
    base_url: str = UVA_API_ENDPOINT,
    model: str = EMBEDDING_MODEL,
    batch_size: int = BATCH_SIZE
) -> int:
    """
    Generate embeddings using UvA's Azure OpenAI proxy — streaming mode (writes per-batch to disk).
    This avoids keeping all chunks/embeddings in memory.
    """

    print("\n" + "="*60)
    print("GENERATING EMBEDDINGS WITH UVA AZURE OPENAI (STREAMING MODE)")
    print("="*60)

    # Initialize client with UvA settings
    client = OpenAI(
        api_key=api_key,
        base_url=base_url
    )

    # Count total chunks (streaming count; does not load content into memory)
    total_chunks = 0
    with open(input_file, "r", encoding="utf-8") as _f:
        for _ in _f:
            total_chunks += 1

    num_batches = (total_chunks + batch_size - 1) // batch_size

    print(f"\nStreaming {total_chunks:,} chunks in ~{num_batches:,} batches (batch_size={batch_size})")
    print(f"Model: {model}")
    print(f"Rate limit delay: {RATE_LIMIT_DELAY}s between batches\n")

    # Statistics
    total_tokens = 0
    total_api_calls = 0
    failed_batches = 0
    processed_chunks = 0

    start_time = time.time()

    # Ensure output directory exists
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    # Open output file in append mode so partial runs are preserved
    out_mode = "a" if Path(output_file).exists() else "w"
    with open(output_file, out_mode, encoding="utf-8") as out_f:

        # Iterate batches from the streaming generator
        for batch_idx, batch in enumerate(tqdm(stream_jsonl_batches(input_file, batch_size),
                                               total=num_batches, desc="Generating embeddings", unit="batch"), start=1):
            # Extract texts
            texts = [chunk["text"] for chunk in batch]

            # Token count (estimate)
            batch_tokens = sum(count_tokens_estimate(text) for text in texts)
            total_tokens += batch_tokens

            try:
                # API CALL to UvA Azure proxy
                response = client.embeddings.create(
                    model=model,
                    input=texts
                )

                total_api_calls += 1

                # Sanity check
                if not hasattr(response, "data") or len(response.data) != len(texts):
                    raise RuntimeError(f"Unexpected response size: {len(getattr(response,'data',[]))} != {len(texts)}")

                # Immediately write each chunk (with embedding) to disk, then remove embedding from memory
                for i, chunk in enumerate(batch):
                    embedding = response.data[i].embedding
                    chunk["embedding"] = embedding
                    out_f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    # remove the embedding to free memory in this local object
                    del chunk["embedding"]
                    processed_chunks += 1

                # flush occasionally to ensure partial progress is on disk
                out_f.flush()

                # Rate limiting wait
                time.sleep(RATE_LIMIT_DELAY)

                # Free batch-level objects and prompt GC
                del response, batch, texts
                gc.collect()

            except Exception as e:
                failed_batches += 1
                error_str = str(e)
                print(f"\n⚠️  Error in batch {batch_idx}: {error_str[:200]}")

                # Write failed items with embedding=None so it's clear which failed
                for chunk in batch:
                    chunk["embedding"] = None
                    out_f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    del chunk["embedding"]
                    processed_chunks += 1

                out_f.flush()
                del batch, texts
                gc.collect()

                # If rate limited, wait a bit longer (simple handling)
                if "rate" in error_str.lower() or "quota" in error_str.lower():
                    print("    Rate limit/quota hit. Waiting 30 seconds...")
                    time.sleep(3)

    elapsed_time = time.time() - start_time

    # Output file size
    output_size = Path(output_file).stat().st_size / (1024 * 1024)

    # Statistics summary (note: processed_chunks equals total lines written)
    print("\n" + "="*60)
    print("EMBEDDING GENERATION COMPLETE (STREAMING)")
    print("="*60)
    print(f"✓ Total chunks processed: {processed_chunks:,}")
    print(f"  Failed batches: {failed_batches}")
    print(f"\n✓ API Statistics:")
    print(f"  Total API calls: {total_api_calls}")
    print(f"  Total tokens (estimate): {total_tokens:,}")
    print(f"\n✓ Output:")
    print(f"  File: {output_file}")
    print(f"  Size: {output_size:.2f} MB")
    print(f"\n✓ Performance:")
    print(f"  Time: {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
    if elapsed_time > 0:
        print(f"  Speed: {processed_chunks/elapsed_time:.2f} chunks/sec")

    return processed_chunks


# ============================================
# VERIFICATION
# ============================================

def verify_embeddings(filepath: str = OUTPUT_FILE) -> None:
    """Verify embeddings file."""
    print("\n" + "="*60)
    print("VERIFYING EMBEDDINGS")
    print("="*60)

    chunks = load_jsonl(filepath)

    with_embeddings = [c for c in chunks if c.get("embedding")]
    without_embeddings = [c for c in chunks if not c.get("embedding")]

    print(f"\nTotal chunks: {len(chunks)}")
    print(f"  With embeddings: {len(with_embeddings)}")
    print(f"  Without embeddings: {len(without_embeddings)}")

    if with_embeddings:
        first_embedding = with_embeddings[0]["embedding"]
        print(f"\nEmbedding dimensions: {len(first_embedding)}")

        print("\n" + "="*60)
        print("SAMPLE CHUNKS")
        print("="*60)

        for i, chunk in enumerate(with_embeddings[:2]):
            print(f"\n--- Sample {i+1} ---")
            print(f"Drug: {chunk['metadata']['drug_name_brand']}")
            print(f"Category: {chunk['metadata']['category']}")
            print(f"Text: {chunk['text'][:100]}...")
            print(f"Embedding (first 5): {chunk['embedding'][:5]}")

    if without_embeddings:
        print(f"\n⚠️  {len(without_embeddings)} chunks failed")


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("UVA AZURE OPENAI EMBEDDINGS GENERATOR")
    print("="*60)

    # Check API access
    print("\n[STEP 1] Checking UvA Azure API access...")
    api_valid = check_available_embedding_models(UVA_API_KEY, UVA_API_ENDPOINT)

    if not api_valid:
        print("\n❌ Cannot proceed without valid API access.")
        exit(1)

    # Confirm
    print("\n" + "="*60)
    proceed = input("\nProceed with embedding generation? (y/n): ").lower().strip()

    if proceed != 'y':
        print("Cancelled.")
        exit(0)

    # Generate
    print("\n[STEP 2] Generating embeddings...")
    num_processed = generate_embeddings_openai_uva()

    # # Verify
    # print("\n[STEP 3] Verifying...")
    # verify_embeddings()

    print("\n" + "="*60)
    print("✓ COMPLETE!")
    print("="*60)
    print(f"✓ File: {OUTPUT_FILE}")
    print("\nNext: Upload to Qdrant (STEP 3)")
