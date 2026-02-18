"""
STEP 1: CHUNKING - WHOLE DATASET VERSION
Transforms OpenFDA filtered data into semantically coherent chunks
optimized for vector search with metadata filtering.

Input: combined_whole.json (ALL drugs from FDA dataset)
Output: chunks_whole.jsonl (ready for embedding)
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from tqdm import tqdm


# ============================================
# CONFIGURATION
# ============================================

# Chunking parameters
MAX_CHUNK_SIZE = 500  # characters
OVERLAP = 50  # characters for context preservation

# Input/Output paths
INPUT_FILE = "combined_whole.json"  # All drugs data
OUTPUT_FILE = "chunks_whole.jsonl"


# ============================================
# HELPER FUNCTIONS
# ============================================

def extract_brand_name(drug: Dict[str, Any]) -> Optional[str]:
    """
    Extract brand name from openfda or fallback to other fields.
    """
    # Try openfda first
    if "openfda" in drug and drug["openfda"]:
        if "brand_name" in drug["openfda"] and drug["openfda"]["brand_name"]:
            return drug["openfda"]["brand_name"][0]

    # Fallback: Try to parse from spl_product_data_elements
    if "spl_product_data_elements" in drug and drug["spl_product_data_elements"]:
        # First element usually contains brand name at the start
        first_element = drug["spl_product_data_elements"][0]
        # Extract first word/phrase (before first space or comma)
        brand = first_element.split()[0] if first_element else None
        if brand:
            return brand.strip()

    # Last resort: Use set_id as identifier
    return f"Drug_{drug.get('set_id', 'unknown')[:8]}"


def extract_generic_names_and_class(drug: Dict[str, Any]) -> tuple[List[str], Optional[str]]:
    """
    Extract generic names and drug class from available fields.

    Returns:
        (generic_names, drug_class)
    """
    generic_names = []
    drug_class = None

    # Try openfda first
    if "openfda" in drug and drug["openfda"]:
        if "generic_name" in drug["openfda"] and drug["openfda"]["generic_name"]:
            openfda_generics = drug["openfda"]["generic_name"]
            if openfda_generics:
                generic_names.extend(openfda_generics)

        # Try to get drug class from pharm_class fields
        if "pharm_class_epc" in drug["openfda"] and drug["openfda"]["pharm_class_epc"]:
            drug_class = drug["openfda"]["pharm_class_epc"][0]
        elif "pharm_class_cs" in drug["openfda"] and drug["openfda"]["pharm_class_cs"]:
            drug_class = drug["openfda"]["pharm_class_cs"][0]

    # Fallback: Try substance_name
    if not generic_names and "openfda" in drug and drug["openfda"]:
        if "substance_name" in drug["openfda"] and drug["openfda"]["substance_name"]:
            generic_names.extend(drug["openfda"]["substance_name"][:3])  # Limit to top 3

    # If still no generic names, use brand name as fallback
    if not generic_names:
        brand = extract_brand_name(drug)
        generic_names = [brand] if brand else ["Unknown"]

    # If still no drug class, try purpose field
    if not drug_class and "purpose" in drug and drug["purpose"]:
        drug_class = drug["purpose"][0]

    return generic_names, drug_class


def create_searchable_variations(brand_name: str, generic_names: List[str]) -> List[str]:
    """
    Create searchable variations for fuzzy matching.
    """
    variations = set()

    # Add original names
    variations.add(brand_name.lower())
    for generic in generic_names:
        variations.add(generic.lower())

    # Add without spaces
    variations.add(brand_name.lower().replace(" ", ""))
    for generic in generic_names:
        variations.add(generic.lower().replace(" ", ""))

    # Add without hyphens
    variations.add(brand_name.lower().replace("-", ""))
    for generic in generic_names:
        variations.add(generic.lower().replace("-", ""))

    # Add combined brand+generic (sometimes users say both)
    for generic in generic_names:
        variations.add(f"{brand_name.lower()}{generic.lower()}")

    return list(variations)


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using basic rules.
    Handles periods, question marks, exclamation marks.
    """
    # Basic sentence splitting (period followed by space and capital letter)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text_smartly(text: str, max_size: int = MAX_CHUNK_SIZE, overlap: int = OVERLAP) -> List[str]:
    """
    Intelligently chunk text by sentences while respecting size limits.
    Adds overlap between chunks for context preservation.

    Args:
        text: Text to chunk
        max_size: Maximum characters per chunk
        overlap: Characters to overlap between chunks

    Returns:
        List of text chunks
    """
    # If text is short enough, return as single chunk
    if len(text) <= max_size:
        return [text]

    # Split into sentences
    sentences = split_into_sentences(text)

    # If splitting failed or only one sentence, do character-based split
    if len(sentences) <= 1:
        chunks = []
        start = 0
        while start < len(text):
            end = start + max_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap  # Overlap for context
        return chunks

    # Build chunks sentence by sentence
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        # If adding this sentence exceeds limit, save current chunk and start new one
        if current_chunk and len(current_chunk) + len(sentence) + 1 > max_size:
            chunks.append(current_chunk.strip())

            # Start new chunk with overlap (last ~overlap chars from previous chunk)
            if overlap > 0 and len(current_chunk) > overlap:
                current_chunk = current_chunk[-overlap:] + " " + sentence
            else:
                current_chunk = sentence
        else:
            # Add sentence to current chunk
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence

    # Don't forget the last chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def create_chunks_for_drug(drug: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create all chunks for a single drug record.

    Process:
    1. Extract drug identifiers (brand, generic, class)
    2. For each field in the drug record:
       - Chunk the field if it's too large
       - Create chunk objects with metadata
    3. Return list of all chunks for this drug
    """
    chunks = []

    # Extract drug identifiers
    set_id = drug.get("set_id", "unknown")
    brand_name = extract_brand_name(drug)
    generic_names, drug_class = extract_generic_names_and_class(drug)

    # Create searchable variations
    searchable_variations = create_searchable_variations(brand_name, generic_names)

    # Process all fields
    # Skip metadata and structural fields
    skip_fields = {"set_id", "id", "effective_time", "version", "openfda", "meta"}

    for field_name, field_value in drug.items():
        # Skip non-content fields
        if field_name in skip_fields:
            continue

        # Skip if empty
        if not field_value:
            continue

        # FDA data usually has fields as lists of strings
        if isinstance(field_value, list):
            # Combine list elements into single text
            field_text = " ".join(str(item) for item in field_value)
        elif isinstance(field_value, str):
            field_text = field_value
        else:
            # Skip non-text fields
            continue

        # Skip very short or empty fields
        if len(field_text.strip()) < 10:
            continue

        # Chunk the field text
        field_chunks = chunk_text_smartly(field_text, MAX_CHUNK_SIZE, OVERLAP)

        # Create chunk objects
        for idx, chunk_text in enumerate(field_chunks):
            chunk = {
                "id": f"{set_id}_{field_name}_{idx}",
                "text": chunk_text,
                "metadata": {
                    "drug_name_brand": brand_name,
                    "drug_name_generic": generic_names[0] if generic_names else "Unknown",
                    "drug_names_all_generics": generic_names,  # All generic names
                    "drug_names_searchable": searchable_variations,
                    "drug_class": drug_class if drug_class else "Unknown",
                    "category": field_name,  # Field type (warnings, dosage, etc.)
                    "set_id": set_id,
                    "chunk_index": idx,
                    "total_chunks_in_category": len(field_chunks)
                }
            }
            chunks.append(chunk)

    return chunks


def save_jsonl(data: List[Dict], filepath: str) -> None:
    """Save data to JSONL format (one JSON object per line)."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def load_json(filepath: str) -> Dict:
    """Load JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


# ============================================
# MAIN CHUNKING FUNCTION
# ============================================

def create_chunks(input_file: str = INPUT_FILE, output_file: str = OUTPUT_FILE) -> int:
    """
    Main function: Process OpenFDA data and create chunks.

    Args:
        input_file: Path to combined_whole.json
        output_file: Path to output chunks_whole.jsonl

    Returns:
        Number of chunks created
    """
    print(f"Loading data from {input_file}...")
    data = load_json(input_file)

    # Extract drug results
    if "results" in data:
        drugs = data["results"]
        print(f"Found {len(drugs)} drug records")
    else:
        drugs = [data]  # Single drug object
        print("Processing single drug record")

    # Process each drug and collect chunks
    all_chunks = []

    print("\nCreating chunks...")
    for drug in tqdm(drugs, desc="Processing drugs"):
        drug_chunks = create_chunks_for_drug(drug)
        all_chunks.extend(drug_chunks)

    # Save to JSONL
    print(f"\nSaving {len(all_chunks)} chunks to {output_file}...")
    save_jsonl(all_chunks, output_file)

    # Statistics
    print("\n" + "="*50)
    print("CHUNKING COMPLETE")
    print("="*50)
    print(f"Input drugs: {len(drugs)}")
    print(f"Output chunks: {len(all_chunks)}")
    print(f"Average chunks per drug: {len(all_chunks) / len(drugs):.1f}")
    print(f"Output file: {output_file}")
    print(f"Output size: {Path(output_file).stat().st_size / 1024 / 1024:.2f} MB")

    return len(all_chunks)


# ============================================
# OPTIONAL: ANALYSIS FUNCTIONS
# ============================================

def analyze_chunks(chunks_file: str = OUTPUT_FILE) -> None:
    """
    Analyze the created chunks to verify quality.
    Useful for debugging and optimization.
    """
    print(f"\nAnalyzing chunks from {chunks_file}...")

    chunks = []
    with open(chunks_file, 'r', encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))

    # Statistics
    chunk_sizes = [len(chunk["text"]) for chunk in chunks]
    categories = {}
    drugs = set()

    for chunk in chunks:
        category = chunk["metadata"]["category"]
        categories[category] = categories.get(category, 0) + 1
        drugs.add(chunk["metadata"]["drug_name_brand"])

    print("\n" + "="*50)
    print("CHUNK ANALYSIS")
    print("="*50)
    print(f"Total chunks: {len(chunks)}")
    print(f"Unique drugs: {len(drugs)}")
    print(f"\nChunk size statistics:")
    print(f"  Min: {min(chunk_sizes)} chars")
    print(f"  Max: {max(chunk_sizes)} chars")
    print(f"  Average: {sum(chunk_sizes) / len(chunk_sizes):.1f} chars")
    print(f"\nChunks by category (top 10):")
    for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {category}: {count}")

    # Sample chunks
    print("\n" + "="*50)
    print("SAMPLE CHUNKS (first 3)")
    print("="*50)
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ---")
        print(f"ID: {chunk['id']}")
        print(f"Drug: {chunk['metadata']['drug_name_brand']} ({chunk['metadata']['drug_name_generic']})")
        print(f"Class: {chunk['metadata']['drug_class']}")
        print(f"Category: {chunk['metadata']['category']}")
        print(f"Text ({len(chunk['text'])} chars): {chunk['text'][:200]}...")


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Create chunks
    num_chunks = create_chunks()

    # Analyze results
    analyze_chunks()

    print("\n✓ Chunking complete!")
    print(f"✓ Created {num_chunks} chunks")
    print(f"✓ Saved to: {OUTPUT_FILE}")
    print("\nNext step: Generate embeddings (STEP 2)")

