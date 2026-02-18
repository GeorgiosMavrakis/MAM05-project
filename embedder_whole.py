"""
STEP 2: SENTENCE-TRANSFORMERS - OPTIMIZED FOR SPEED - WHOLE DATASET VERSION
GPU-accelerated with parallel I/O processing.
"""

import json
import time
from pathlib import Path
from typing import Iterator, Dict, List
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch


# ============================================
# CONFIGURATION
# ============================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
INPUT_FILE = "chunks_whole.jsonl"
OUTPUT_FILE = "chunks_with_embeddings_whole.jsonl"

# OPTIMIZED SETTINGS FOR SPEED
BATCH_SIZE = 10000  # Process 10k at a time
ENCODING_BATCH = 512  # Much larger batch for GPU
FLOAT_PRECISION = 4  # 4 decimals is enough (saves space and time)


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


def stream_jsonl_batch(filepath: str, batch_size: int) -> Iterator[List[Dict]]:
    """Stream JSONL in batches for faster processing."""
    batch = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            batch.append(json.loads(line))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def write_jsonl_batch(data_list: List[Dict], filepath: str) -> None:
    """Write batch to file efficiently."""
    with open(filepath, 'a', encoding='utf-8') as f:
        for data in data_list:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')


def round_embedding(embedding: np.ndarray, precision: int) -> list:
    """Round numpy array to list with precision."""
    return [round(float(x), precision) for x in embedding]


def detect_device() -> str:
    """Detect if GPU is available."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"



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
    Generate embeddings with GPU acceleration and parallel I/O.
    """

    print("\n" + "="*60)
    print("LOCAL EMBEDDINGS (GPU ACCELERATED) - WHOLE DATASET")
    print("="*60)

    # Check CUDA availability first
    print("\n🔍 Checking GPU availability...")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA not available!")
        print("To enable GPU acceleration:")
        print("  1. Check if you have an NVIDIA GPU")
        print("  2. Install CUDA drivers")
        print("  3. Install PyTorch with CUDA:")
        print("     pip uninstall torch")
        print("     pip install torch --index-url https://download.pytorch.org/whl/cu118")
        print("\nProceeding with CPU (will be MUCH slower)...\n")
        input("Press Enter to continue with CPU...")

    # Detect device
    device = detect_device()
    print(f"\n✓ Device: {device.upper()}")

    if device == "cuda":
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
        print(f"✓ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"✓ CUDA Version: {torch.version.cuda}")

    # Load model once and FORCE it to GPU
    print(f"\nLoading model: {model_name}...")
    model = SentenceTransformer(model_name)

    # Explicitly move to device
    if device == "cuda":
        print("Moving model to GPU...")
        model = model.to('cuda')
        model = model.half()  # Use FP16 for speed and memory
        print("✓ Model on GPU with FP16 (half precision)")
    else:
        model = model.to('cpu')

    print("✓ Model loaded")

    # Adjust batch sizes for GPU
    if device == "cuda":
        # GPU can handle much larger batches
        encoding_batch = 2048  # Very large batch for GPU
        file_batch = batch_size
        print(f"\n⚡ GPU optimizations enabled:")
        print(f"   File batch: {file_batch:,}")
        print(f"   Encoding batch: {encoding_batch:,}")
    else:
        encoding_batch = ENCODING_BATCH
        file_batch = batch_size
        print(f"\nCPU mode - smaller batches:")
        print(f"   File batch: {file_batch:,}")
        print(f"   Encoding batch: {encoding_batch}")

    # Count total
    total_chunks = count_lines(input_file)
    print(f"\nTotal chunks: {total_chunks:,}")
    print(f"Float precision: {FLOAT_PRECISION} decimals\n")

    # Clear output
    if Path(output_file).exists():
        Path(output_file).unlink()

    # Process in batches
    processed = 0
    start_time = time.time()

    with tqdm(total=total_chunks, desc="Processing", unit="chunks") as pbar:
        for batch_chunks in stream_jsonl_batch(input_file, file_batch):
            # Extract texts
            batch_texts = [chunk["text"] for chunk in batch_chunks]

            # Encode batch - model is already on the correct device
            # Explicitly pass device to ensure GPU usage
            embeddings = model.encode(
                batch_texts,
                batch_size=encoding_batch,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True,
                device=device  # Force GPU usage
            )

            # Attach embeddings
            for i, chunk in enumerate(batch_chunks):
                chunk["embedding"] = round_embedding(
                    embeddings[i], FLOAT_PRECISION
                )

            # Write batch to file
            write_jsonl_batch(batch_chunks, output_file)

            processed += len(batch_chunks)
            pbar.update(len(batch_chunks))

            # Show speed estimate
            elapsed = time.time() - start_time
            speed = processed / elapsed
            remaining = (total_chunks - processed) / speed if speed > 0 else 0
            pbar.set_postfix({
                'speed': f'{speed:.0f} c/s',
                'ETA': f'{remaining/60:.1f}m'
            })

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

    for batch in stream_jsonl_batch(filepath, 1000):
        for chunk in batch:
            total += 1
            if first_emb is None and chunk.get("embedding"):
                first_emb = chunk["embedding"]
                sample = chunk
        if first_emb:
            break

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
    print("SENTENCE-TRANSFORMERS (MEMORY EFFICIENT) - WHOLE DATASET")
    print("="*60)

    proceed = input("\nProceed? (y/n): ").lower().strip()
    if proceed != 'y':
        exit(0)

    generate_embeddings_local_efficient()
    verify_embeddings_efficient()

    print("\n✓ Next: Upload to Qdrant")
