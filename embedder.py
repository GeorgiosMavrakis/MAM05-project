"""
STEP 2: SENTENCE-TRANSFORMERS - MEMORY EFFICIENT
Processes in chunks, never loads full file.
"""

import json
import time
from pathlib import Path
from typing import Iterator, Dict
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# ============================================
# CONFIGURATION
# ============================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
INPUT_FILE = "chunks.jsonl"
OUTPUT_FILE = "chunks_with_embeddings.jsonl"

BATCH_SIZE = 1000  # Process 1000 at a time (adjust based on RAM)
ENCODING_BATCH = 32  # Model's internal batch size
FLOAT_PRECISION = 4  # Decimal places


# ============================================
# MEMORY-EFFICIENT HELPERS
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


def append_jsonl(data: Dict, filepath: str) -> None:
    """Append to file."""
    with open(filepath, 'a', encoding='utf-8') as f:
        f.write(json.dumps(data, ensure_ascii=False) + '\n')


def round_embedding(embedding: np.ndarray, precision: int) -> list:
    """Round numpy array to list with precision."""
    return [round(float(x), precision) for x in embedding]


# ============================================
# MEMORY-EFFICIENT GENERATION
# ============================================

def generate_embeddings_local_efficient(
    input_file: str = INPUT_FILE,
    output_file: str = OUTPUT_FILE,
    model_name: str = EMBEDDING_MODEL,
    batch_size: int = BATCH_SIZE
) -> int:
    """
    Generate embeddings with streaming - memory efficient.
    """

    print("\n" + "="*60)
    print("LOCAL EMBEDDINGS (MEMORY EFFICIENT)")
    print("="*60)

    # Load model once
    print(f"\nLoading model: {model_name}...")
    model = SentenceTransformer(model_name)
    print("✓ Model loaded")

    # Count total
    total_chunks = count_lines(input_file)
    print(f"\nTotal chunks: {total_chunks:,}")
    print(f"Processing {batch_size} at a time")
    print(f"Float precision: {FLOAT_PRECISION} decimals\n")

    # Clear output
    if Path(output_file).exists():
        Path(output_file).unlink()

    # Process in batches
    processed = 0
    start_time = time.time()

    batch_chunks = []
    batch_texts = []

    with tqdm(total=total_chunks, desc="Processing") as pbar:

        for chunk in stream_jsonl(input_file):
            batch_chunks.append(chunk)
            batch_texts.append(chunk["text"])

            # When batch full, process
            if len(batch_chunks) >= batch_size:
                # Encode batch
                embeddings = model.encode(
                    batch_texts,
                    batch_size=ENCODING_BATCH,
                    show_progress_bar=False,
                    normalize_embeddings=True,
                    convert_to_numpy=True
                )

                # Attach and save
                for i, chunk in enumerate(batch_chunks):
                    chunk["embedding"] = round_embedding(
                        embeddings[i], FLOAT_PRECISION
                    )
                    append_jsonl(chunk, output_file)

                processed += len(batch_chunks)
                pbar.update(len(batch_chunks))

                # Clear batch
                batch_chunks = []
                batch_texts = []

        # Process remaining
        if batch_chunks:
            embeddings = model.encode(
                batch_texts,
                batch_size=ENCODING_BATCH,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True
            )

            for i, chunk in enumerate(batch_chunks):
                chunk["embedding"] = round_embedding(
                    embeddings[i], FLOAT_PRECISION
                )
                append_jsonl(chunk, output_file)

            processed += len(batch_chunks)
            pbar.update(len(batch_chunks))

    elapsed = time.time() - start_time
    size_gb = Path(output_file).stat().st_size / (1024**3)

    print("\n" + "="*60)
    print("COMPLETE")
    print("="*60)
    print(f"✓ Processed: {processed:,}")
    print(f"✓ Output: {output_file}")
    print(f"  Size: {size_gb:.2f} GB")
    print(f"✓ Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"✓ Speed: {processed/elapsed:.0f} chunks/sec")

    return processed


# ============================================
# VERIFICATION
# ============================================

def verify_embeddings_efficient(filepath: str = OUTPUT_FILE) -> None:
    """Verify without loading all."""
    print("\n" + "="*60)
    print("VERIFYING")
    print("="*60)

    total = 0
    first_emb = None
    sample = None

    for chunk in stream_jsonl(filepath):
        total += 1
        if first_emb is None and chunk.get("embedding"):
            first_emb = chunk["embedding"]
            sample = chunk

    print(f"\n✓ Total: {total:,}")
    if first_emb:
        print(f"✓ Dimensions: {len(first_emb)}")
        print(f"\nSample:")
        print(f"  Drug: {sample['metadata']['drug_name_brand']}")
        print(f"  Embedding: {first_emb[:5]}")

    size_gb = Path(filepath).stat().st_size / (1024**3)
    print(f"\n✓ Size: {size_gb:.2f} GB")


# ============================================
# MAIN
# ============================================

if __name__ == "__main__":
    print("="*60)
    print("SENTENCE-TRANSFORMERS (MEMORY EFFICIENT)")
    print("="*60)

    proceed = input("\nProceed? (y/n): ").lower().strip()
    if proceed != 'y':
        exit(0)

    generate_embeddings_local_efficient()
    verify_embeddings_efficient()

    print("\n✓ Next: Upload to Qdrant")
