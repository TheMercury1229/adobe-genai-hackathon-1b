"""
Script to parse input.json, load outline JSONs, and create mapping between PDF filenames and outlines.
Updated version with configurable collection folder path.
"""

import json
import os
import glob


def load_outline_files(collection_folder):
    """
    Load all outline JSON files from the specified collection folder's JSON subfolder.
    Returns a dictionary mapping filenames to their outline structures.
    
    Args:
        collection_folder (str): Path to the collection folder (e.g., "../Collection1")
    """
    outline_folder = os.path.join(collection_folder, "JSON")
    print(f"Loading outline files from: {outline_folder}")
    outline_data = {}
    outline_files = glob.glob(os.path.join(outline_folder, "*.json"))

    print(f"Looking for outline files in: {outline_folder}")
    print(
        f"Found {len(outline_files)} JSON files: {[os.path.basename(f) for f in outline_files]}")

    for outline_file in outline_files:
        try:
            with open(outline_file, 'r', encoding='utf-8') as file:
                outline = json.load(file)

            # Extract filename from the outline JSON
            if "filename" in outline:
                pdf_filename = outline["filename"]
                outline_data[pdf_filename] = outline
                print(f"  Loaded outline for: {pdf_filename}")
            else:
                print(f"Warning: No 'filename' key found in {outline_file}")
                # Debug: print the keys that are available
                print(f"  Available keys: {list(outline.keys())}")

        except FileNotFoundError:
            print(f"Error: Outline file {outline_file} not found.")
        except KeyError as e:
            print(f"Error: Missing key in outline JSON {outline_file} - {e}")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in {outline_file}")

    return outline_data


def process_json_with_outlines(collection_folder, input_filename="challenge1b_input.json"):
    """
    Load JSON file, extract information, and create mapping to outline structures.
    
    Args:
        collection_folder (str): Path to the collection folder (e.g., "../Collection1")
        input_filename (str): Name of the input JSON file (default: "challenge1b_input.json")
    """
    input_path = os.path.join(collection_folder, input_filename)
    
    try:
        # Load and parse input.json
        with open(input_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        # Extract required information
        persona = data["persona"]["role"]
        job = data["job_to_be_done"]["task"]
        pdf_filenames = [doc["filename"] for doc in data["documents"]]

        # Load outline files
        print(f"\nLoading outline files from: {collection_folder}")
        outline_data = load_outline_files(collection_folder)
        print(f"Loaded {len(outline_data)} outline files")
        print(f"Outline filenames: {list(outline_data.keys())}")
        print(f"PDF filenames from input.json: {pdf_filenames}")

        # Create mapping between PDF filenames and their outlines
        pdf_to_outline_mapping = {}
        matched_outlines = set()

        for pdf_filename in pdf_filenames:
            if pdf_filename in outline_data:
                pdf_to_outline_mapping[pdf_filename] = outline_data[pdf_filename]
                matched_outlines.add(pdf_filename)
            else:
                print(f"Warning: No outline found for PDF: {pdf_filename}")

        # Check for unmatched outline files
        unmatched_outlines = set(outline_data.keys()) - matched_outlines
        if unmatched_outlines:
            print(
                f"Warning: Outline files found but not in input.json: {list(unmatched_outlines)}")

        # Print results
        print(f"Persona: {persona}")
        print(f"Job to be done: {job}")
        print(f"Document filenames: {pdf_filenames}")
        print(
            f"Successfully mapped {len(pdf_to_outline_mapping)} PDFs to outlines")
        print(
            f"Unmatched PDFs: {len(pdf_filenames) - len(pdf_to_outline_mapping)}")
        print(f"Unmatched outline files: {len(unmatched_outlines)}")

        return persona, job, pdf_filenames, pdf_to_outline_mapping

    except FileNotFoundError:
        print(f"Error: {input_path} file not found.")
        return None, None, None, None
    except KeyError as e:
        print(f"Error: Missing key in JSON - {e}")
        return None, None, None, None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {input_path}")
        return None, None, None, None


def print_mapping_summary(pdf_to_outline_mapping):
    """
    Print a summary of the PDF to outline mapping.
    """
    print("\n" + "="*50)
    print("PDF TO OUTLINE MAPPING SUMMARY")
    print("="*50)

    for pdf_filename, outline in pdf_to_outline_mapping.items():
        print(f"\nPDF: {pdf_filename}")
        if "title" in outline:
            print(f"  Title: {outline['title']}")
        if "sections" in outline:
            print(f"  Sections: {len(outline['sections'])}")
        if "outline_structure" in outline:
            print(f"  Has outline structure: Yes")


if __name__ == "__main__":
    # Default collection folder for standalone execution
    default_collection = "../Collection1"
    
    # Process the JSON and create mappings
    persona, job, filenames, mapping = process_json_with_outlines(default_collection)

    if mapping:
        print_mapping_summary(mapping)