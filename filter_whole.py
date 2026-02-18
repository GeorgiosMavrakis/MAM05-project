"""
JSON Filtering Functions for FDA Data - WHOLE DATASET VERSION

This module provides functions to filter and clean FDA OpenFDA JSON data files.
This version processes ALL drugs (not just diabetic medications).
"""

import json
import re
from typing import Dict, List, Any, Union
from pathlib import Path


def remove_table_keys(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Remove all key-value pairs from result dictionaries where the key contains '_table'.

    Args:
        data: Dictionary containing the full JSON structure with 'meta' and 'results'

    Returns:
        Modified dictionary with '_table' keys removed from all results
    """
    # Create a deep copy to avoid modifying the original
    filtered_data = json.loads(json.dumps(data))

    if 'results' not in filtered_data:
        return filtered_data

    for result in filtered_data['results']:
        if not isinstance(result, dict):
            continue

        # Find all keys containing '_table'
        keys_to_remove = [key for key in result.keys() if '_table' in key]

        # Remove those keys
        for key in keys_to_remove:
            del result[key]

    return filtered_data


def filter_openfda_keys(data: Dict[str, Any], regex_patterns: List[str]) -> Dict[str, Any]:
    """
    Filter the 'openfda' nested dictionary to keep only keys matching specified regex patterns.

    Args:
        data: Dictionary containing the full JSON structure with 'meta' and 'results'
        regex_patterns: List of regex pattern strings to match against openfda keys

    Returns:
        Modified dictionary with filtered openfda dictionaries
    """
    # Create a deep copy to avoid modifying the original
    filtered_data = json.loads(json.dumps(data))

    if 'results' not in filtered_data:
        return filtered_data

    # Compile regex patterns for efficiency
    compiled_patterns = [re.compile(pattern) for pattern in regex_patterns]

    for result in filtered_data['results']:
        if not isinstance(result, dict):
            continue

        if 'openfda' not in result or not isinstance(result['openfda'], dict):
            continue

        openfda = result['openfda']

        # Find keys that match at least one regex pattern
        keys_to_keep = []
        for key in openfda.keys():
            if any(pattern.search(key) for pattern in compiled_patterns):
                keys_to_keep.append(key)

        # Create new openfda dict with only matching keys
        filtered_openfda = {key: openfda[key] for key in keys_to_keep}
        result['openfda'] = filtered_openfda

    return filtered_data


def remove_keys_by_regex(data: Dict[str, Any], regex_patterns: List[str]) -> Dict[str, Any]:
    """
    Remove key-value pairs from result dictionaries where keys match specified regex patterns.

    Args:
        data: Dictionary containing the full JSON structure with 'meta' and 'results'
        regex_patterns: List of regex pattern strings to match against result keys

    Returns:
        Modified dictionary with matching keys removed from all results
    """
    # Create a deep copy to avoid modifying the original
    filtered_data = json.loads(json.dumps(data))

    if 'results' not in filtered_data:
        return filtered_data

    # Compile regex patterns for efficiency
    compiled_patterns = [re.compile(pattern) for pattern in regex_patterns]

    for result in filtered_data['results']:
        if not isinstance(result, dict):
            continue

        # Find all keys that match at least one regex pattern
        keys_to_remove = []
        for key in result.keys():
            if any(pattern.search(key) for pattern in compiled_patterns):
                keys_to_remove.append(key)

        # Remove those keys
        for key in keys_to_remove:
            del result[key]

    return filtered_data


def process_json_file(
        input_filepath: Union[str, Path],
        output_filepath: Union[str, Path],
        remove_tables: bool = False,
        openfda_regex_patterns: List[str] = None,
        remove_result_keys_patterns: List[str] = None
) -> None:
    """
    Process a JSON file with specified filtering operations.

    Args:
        input_filepath: Path to input JSON file
        output_filepath: Path to output JSON file
        remove_tables: If True, remove all keys containing '_table'
        openfda_regex_patterns: If provided, filter openfda keys by these patterns
        remove_result_keys_patterns: If provided, remove result keys matching these patterns
    """
    # Load the JSON file
    with open(input_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    original_count = len(data.get('results', []))

    # Apply filters in order
    # 1. Remove table keys
    if remove_tables:
        data = remove_table_keys(data)

    # 2. Remove specific result keys
    if remove_result_keys_patterns:
        data = remove_keys_by_regex(data, remove_result_keys_patterns)

    # 3. Filter openfda keys
    if openfda_regex_patterns:
        data = filter_openfda_keys(data, openfda_regex_patterns)

    # Save the filtered data
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    final_count = len(data.get('results', []))
    print(f"Processed {input_filepath.name if hasattr(input_filepath, 'name') else input_filepath} -> {output_filepath.name if hasattr(output_filepath, 'name') else output_filepath} ({final_count} results)")


def process_multiple_files(
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        file_pattern: str = "*.json",
        remove_tables: bool = False,
        openfda_regex_patterns: List[str] = None,
        remove_result_keys_patterns: List[str] = None
) -> None:
    """
    Process multiple JSON files in a directory.

    Args:
        input_dir: Directory containing input JSON files
        output_dir: Directory for output JSON files
        file_pattern: Glob pattern for selecting files (default: "*.json")
        remove_tables: If True, remove all keys containing '_table'
        openfda_regex_patterns: If provided, filter openfda keys by these patterns
        remove_result_keys_patterns: If provided, remove result keys matching these patterns
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    # Process each matching file
    for input_file in sorted(input_path.glob(file_pattern)):
        output_file = output_path / input_file.name
        process_json_file(
            input_file,
            output_file,
            remove_tables=remove_tables,
            openfda_regex_patterns=openfda_regex_patterns,
            remove_result_keys_patterns=remove_result_keys_patterns
        )

    print(f"\n✓ Processed all files from {input_dir} to {output_dir}")


def combine_json_files(
        input_dir: Union[str, Path],
        output_filepath: Union[str, Path],
        file_pattern: str = "*.json"
) -> None:
    """
    Combine multiple JSON files into one, using memory-efficient streaming.
    Only keeps last_updated date and total count in metadata.

    This function processes files one at a time to avoid loading all data into memory,
    making it suitable for very large datasets.

    Args:
        input_dir: Directory containing input JSON files to combine
        output_filepath: Path to output combined JSON file
        file_pattern: Glob pattern for selecting files (default: "*.json")
    """
    input_path = Path(input_dir)
    output_path = Path(output_filepath)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_results = 0
    last_updated = None
    file_count = 0

    print(f"Combining JSON files from {input_dir}...")

    # Open output file and write the beginning
    with open(output_path, 'w', encoding='utf-8') as outfile:
        # Write opening of JSON structure (will update meta later)
        outfile.write('{\n  "meta": ')
        meta_position = outfile.tell()  # Remember position to update later
        outfile.write(' ' * 200)  # Reserve space for meta (will overwrite)
        outfile.write(',\n  "results": [\n')

        first_result = True

        # Process each file
        for input_file in sorted(input_path.glob(file_pattern)):
            file_count += 1
            print(f"Processing {input_file.name}...", end=' ')

            # Read file and extract data
            with open(input_file, 'r', encoding='utf-8') as infile:
                data = json.load(infile)

                # Extract last_updated from first file
                if last_updated is None and 'meta' in data:
                    last_updated = data['meta'].get('last_updated', 'unknown')

                # Write results one by one
                if 'results' in data:
                    results_in_file = len(data['results'])
                    for result in data['results']:
                        if not first_result:
                            outfile.write(',\n')
                        else:
                            first_result = False

                        # Write the result
                        json.dump(result, outfile, ensure_ascii=False)
                        total_results += 1

                    print(f"{results_in_file} results")

        # Close the results array and JSON structure
        outfile.write('\n  ]\n}')

        # Now go back and write the actual meta
        outfile.seek(meta_position)
        meta = {
            "last_updated": last_updated,
            "total_drugs": total_results,
            "source_files": file_count
        }
        meta_str = json.dumps(meta, ensure_ascii=False)
        # Pad to fit in reserved space
        meta_str = meta_str.ljust(200)
        outfile.write(meta_str)

    print(f"\n✓ Combined {file_count} files into {output_filepath}")
    print(f"  Total drugs: {total_results}")
    print(f"  Last updated: {last_updated}")


if __name__ == "__main__":
    # Configuration for whole dataset
    keep_regexes = [r"brand_name", r"generic_name", r"pharm_class", r"route", r"rxcui",
                    r"substance_name"]

    # Process ALL drugs from the Drugs folder
    process_multiple_files(
        'Drugs/',
        'Drugs_filtered/',
        remove_tables=True,
        openfda_regex_patterns=keep_regexes,
        remove_result_keys_patterns=["questions", "clinical_studies", "references"]
    )

    # Combine into single file
    combine_json_files('Drugs_filtered/', 'combined_whole.json', file_pattern='*.json')

