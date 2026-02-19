# import ijson
# import orjson
# import re
# from typing import Iterable, List, Pattern, Optional, Dict, Any
# from pathlib import Path

"""
JSON Filtering Functions for FDA Data

This module provides functions to filter and clean FDA OpenFDA JSON data files.
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

    Example:
        >>> data = {"results": [{"name": "test", "dosage_table": ["data"], "id": "123"}]}
        >>> filtered = remove_table_keys(data)
        >>> # Result: {"results": [{"name": "test", "id": "123"}]}
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

    Example:
        >>> data = {"results": [{"id": "1", "openfda": {"brand_name": ["X"], "route": ["Y"], "other": ["Z"]}}]}
        >>> patterns = ["^brand_", "route"]
        >>> filtered = filter_openfda_keys(data, patterns)
        >>> # Result keeps only brand_name and route in openfda
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

    Example:
        >>> data = {"results": [{"id": "1", "warnings": ["text"], "dosage_table": ["data"]}]}
        >>> patterns = ["warning", "_table"]
        >>> filtered = remove_keys_by_regex(data, patterns)
        >>> # Result: {"results": [{"id": "1"}]}
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


def filter_diabetic_drugs(data: Dict[str, Any], diabetic_drug_list: List[str]) -> Dict[str, Any]:
    """
    Filter results to keep only drugs that match diabetic medication names.

    Args:
        data: Dictionary containing the full JSON structure with 'meta' and 'results'
        diabetic_drug_list: List of diabetic drug names to match (case-insensitive)

    Returns:
        Modified dictionary containing only results matching diabetic drugs

    Logic:
        - If 'openfda' is not empty: check 'generic_name' or 'substance_name'
        - If 'openfda' is empty: check 'spl_product_data_elements' or 'active_ingredient'
        - Matching is case-insensitive and checks for whole word matches
        - Uses optimized word splitting with comma replacement
        - Breaks immediately upon finding a match

    Example:
        >>> diabetic_drugs = ["metformin", "insulin", "glipizide"]
        >>> data = {"results": [...]}
        >>> filtered = filter_diabetic_drugs(data, diabetic_drugs)
    """
    # Create a deep copy to avoid modifying the original
    filtered_data = json.loads(json.dumps(data))

    if 'results' not in filtered_data:
        return filtered_data

    # Convert diabetic drug list to lowercase set for O(1) lookup
    diabetic_drugs_set = set(drug.lower() for drug in diabetic_drug_list)

    filtered_results = []

    for result in filtered_data['results']:
        if not isinstance(result, dict):
            continue

        keep_result = False

        # Check if openfda exists and is not empty
        if 'openfda' in result and isinstance(result['openfda'], dict) and result['openfda']:
            # Check generic_name
            if 'generic_name' in result['openfda']:
                generic_names = result['openfda']['generic_name']
                if isinstance(generic_names, list):
                    for name in generic_names:
                        if isinstance(name, str):
                            # Split into words, replacing commas with spaces
                            words = name.replace(',', ' ').lower().split()
                            if any(word in diabetic_drugs_set for word in words):
                                keep_result = True
                                break
                    # if keep_result:
                    #     break

            # Check substance_name if not found yet
            if not keep_result and 'substance_name' in result['openfda']:
                substance_names = result['openfda']['substance_name']
                if isinstance(substance_names, list):
                    for name in substance_names:
                        if isinstance(name, str):
                            # Split into words, replacing commas with spaces
                            words = name.replace(',', ' ').lower().split()
                            if any(word in diabetic_drugs_set for word in words):
                                keep_result = True
                                break
                    # if keep_result:
                    #     break

        # If openfda is empty or doesn't exist, check alternative fields
        if not keep_result:
            # Check active_ingredient
            if 'active_ingredient' in result:
                ingredients = result['active_ingredient']
                if isinstance(ingredients, list):
                    # Process ALL elements, not just the first one
                    for ingredient in ingredients:
                        if isinstance(ingredient, str):
                            # Split into words, replacing commas with spaces
                            words = ingredient.replace(',', ' ').lower().split()
                            if any(word in diabetic_drugs_set for word in words):
                                keep_result = True
                                break
                    # if keep_result:
                    #     break

            # Check spl_product_data_elements if not found yet
            if not keep_result and 'spl_product_data_elements' in result:
                elements = result['spl_product_data_elements']
                if isinstance(elements, list):
                    # Process ALL elements, not just the first one
                    for element in elements:
                        if isinstance(element, str):
                            # Split into words, replacing commas with spaces
                            words = element.replace(',', ' ').lower().split()
                            if any(word in diabetic_drugs_set for word in words):
                                keep_result = True
                                break
                    # if keep_result:
                    #     break

        if keep_result:
            filtered_results.append(result)

    # Update the results
    filtered_data['results'] = filtered_results

    # Update meta if it exists
    if 'meta' in filtered_data and isinstance(filtered_data['meta'], dict):
        if 'results' in filtered_data['meta'] and isinstance(filtered_data['meta']['results'], dict):
            filtered_data['meta']['results']['total'] = len(filtered_results)

    return filtered_data


def process_json_file(
        input_filepath: Union[str, Path],
        output_filepath: Union[str, Path],
        remove_tables: bool = False,
        openfda_regex_patterns: List[str] = None,
        remove_result_keys_patterns: List[str] = None,
        diabetic_drug_list: List[str] = None
) -> None:
    """
    Process a JSON file with specified filtering operations.

    Args:
        input_filepath: Path to input JSON file
        output_filepath: Path to output JSON file
        remove_tables: If True, remove all keys containing '_table'
        openfda_regex_patterns: If provided, filter openfda keys by these patterns
        remove_result_keys_patterns: If provided, remove result keys matching these patterns
        diabetic_drug_list: If provided, filter to keep only diabetic drugs

    Example:
        >>> diabetic_drugs = ["metformin", "insulin", "glipizide"]
        >>> process_json_file(
        ...     'input.json',
        ...     'output.json',
        ...     remove_tables=True,
        ...     openfda_regex_patterns=['^brand_', 'product_ndc', 'manufacturer'],
        ...     remove_result_keys_patterns=['warnings', 'inactive_ingredient'],
        ...     diabetic_drug_list=diabetic_drugs
        ... )
    """
    # Load the JSON file
    with open(input_filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)

    original_count = len(data.get('results', []))

    # Apply filters in order
    # 1. First filter for diabetic drugs if specified (reduces dataset early)
    if diabetic_drug_list:
        data = filter_diabetic_drugs(data, diabetic_drug_list)
        diabetic_count = len(data.get('results', []))
        print(f"  Diabetic filter: {original_count} -> {diabetic_count} results")

    # 2. Remove table keys
    if remove_tables:
        data = remove_table_keys(data)

    # 3. Remove specific result keys
    if remove_result_keys_patterns:
        data = remove_keys_by_regex(data, remove_result_keys_patterns)

    # 4. Filter openfda keys
    if openfda_regex_patterns:
        data = filter_openfda_keys(data, openfda_regex_patterns)

    # Save the filtered data
    with open(output_filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    final_count = len(data.get('results', []))
    print(f"Processed {input_filepath} -> {output_filepath} ({final_count} results)")


def process_multiple_files(
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        file_pattern: str = "*.json",
        remove_tables: bool = False,
        openfda_regex_patterns: List[str] = None,
        remove_result_keys_patterns: List[str] = None,
        diabetic_drug_list: List[str] = None
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
        diabetic_drug_list: If provided, filter to keep only diabetic drugs

    Example:
        >>> diabetic_drugs = ["metformin", "insulin", "glipizide"]
        >>> process_multiple_files(
        ...     'raw_data/',
        ...     'filtered_data/',
        ...     remove_tables=True,
        ...     openfda_regex_patterns=['^brand_name$', '^product_ndc$'],
        ...     remove_result_keys_patterns=['warnings', 'inactive_ingredient'],
        ...     diabetic_drug_list=diabetic_drugs
        ... )
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
            remove_result_keys_patterns=remove_result_keys_patterns,
            diabetic_drug_list=diabetic_drug_list
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

    Example:
        >>> combine_json_files(
        ...     'filtered_data/',
        ...     'combined.json',
        ...     file_pattern='*.json'
        ... )
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
    # files = [f"/path/to/openfda_{i}.json" for i in range(1, 14)]
    # files = "drug-label-0001-of-0013.json"
    keep_regexes = [r"brand_name", r"generic_name", r"pharm_class", r"route", r"rxcui",
                    r"substance_name"]  # , r"upc", r"nui" # example strings; regex allowed
    DIABETIC_DRUGS = [
        "metformin",
        "semaglutide",
        "tirzepatide",
        "glipizide",
        "glimepiride",
        "pioglitazone",
        "liraglutide",
        "sitagliptin",
        "dapagliflozin",
        "empagliflozin",
        "glyburide",
        "canagliflozin",
        "chromium picolinate",
        "exenatide",
        "insulin",
        "repaglinide",
        "dulaglutide",
        "linagliptin",
        "acarbose",
        "alogliptin",
        "bromocriptine",
        "colesevelam",
        "lixisenatide",
        "saxagliptin",
        "miglitol",
        "pramlintide",
        "albiglutide",
        "bexagliflozin",
        "ertugliflozin",
        "nateglinide",
        "rosiglitazone",
        "sotagliflozin",
        "chlorpropamide",
        "tolbutamide",
        "tolazamide",
    ]
    #
    # process_json_file("drug-label-0001-of-0013.json",'output.json',remove_tables=True,
    #                   openfda_regex_patterns=keep_regexes)
    process_multiple_files('openfda_raw/', 'openfda_filtered/', remove_tables=True,
                           openfda_regex_patterns=keep_regexes,
                           remove_result_keys_patterns=["questions", "clinical_studies", "references"],
                           diabetic_drug_list=DIABETIC_DRUGS)

    combine_json_files('openfda_filtered/', 'combined.json', file_pattern='*.json')
