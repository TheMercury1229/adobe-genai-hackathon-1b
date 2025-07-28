"""
Multi-Collection PDF Processing Pipeline
Processes all Collection folders and generates challenge output JSON files
"""

import os
import json
import glob
import sys
from typing import List, Dict, Any, Tuple

# Import all the processing modules
from process_and_map_json import load_outline_files
from extract_text import process_pdfs_with_outlines
from embedder import search_pdf_sections
from summariser import create_refined_summaries


def find_collection_folders(root_dir: str = ".") -> List[str]:
    """
    Find all folders starting with 'Collection' in the root directory.
    
    Args:
        root_dir (str): Root directory to search in
        
    Returns:
        List[str]: List of collection folder paths
    """
    collection_pattern = os.path.join(root_dir, "Collection*")
    collection_folders = glob.glob(collection_pattern)
    collection_folders = [folder for folder in collection_folders if os.path.isdir(folder)]
    
    print(f"Found {len(collection_folders)} Collection folders:")
    for folder in collection_folders:
        print(f"  - {folder}")
    
    return collection_folders


def load_input_json(collection_folder: str) -> Tuple[str, str, List[str], str]:
    """
    Load the input JSON file from collection folder.
    
    Args:
        collection_folder (str): Path to collection folder
        
    Returns:
        Tuple: (persona, job, pdf_filenames, input_filename)
    """
    # Look for challenge input files
    input_patterns = [
        os.path.join(collection_folder, "challenge*_input.json"),
        os.path.join(collection_folder, "*_input.json"),
        os.path.join(collection_folder, "input.json")
    ]
    
    input_file = None
    for pattern in input_patterns:
        matches = glob.glob(pattern)
        if matches:
            input_file = matches[0]  # Take the first match
            break
    
    if not input_file:
        raise FileNotFoundError(f"No input JSON file found in {collection_folder}")
    
    print(f"Loading input from: {input_file}")
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        persona = data["persona"]["role"]
        job = data["job_to_be_done"]["task"]
        pdf_filenames = [doc["filename"] for doc in data["documents"]]
        
        return persona, job, pdf_filenames, os.path.basename(input_file)
        
    except Exception as e:
        raise Exception(f"Error loading input JSON {input_file}: {e}")


def process_single_collection(collection_folder: str) -> bool:
    """
    Process a single collection folder through the complete pipeline.
    
    Args:
        collection_folder (str): Path to collection folder
        
    Returns:
        bool: True if successful, False otherwise
    """
    collection_name = os.path.basename(collection_folder)
    print(f"\n{'='*80}")
    print(f"PROCESSING COLLECTION: {collection_name}")
    print(f"{'='*80}")
    
    try:
        # Step 1: Load input JSON
        print("\n=== Step 1: Loading input JSON ===")
        persona, job, pdf_filenames, input_filename = load_input_json(collection_folder)
        
        print(f"Persona: {persona}")
        print(f"Job: {job}")
        print(f"PDF files to process: {len(pdf_filenames)}")
        
        # Step 2: Load outline files
        print("\n=== Step 2: Loading outline files ===")
        json_folder = os.path.join(collection_folder, "JSON")
        outline_data = load_outline_files(json_folder)
        
        if not outline_data:
            print(f"Warning: No outline files found in {json_folder}")
            return False
        
        # Create mapping between PDFs and outlines
        pdf_to_outline_mapping = {}
        matched_count = 0
        
        for pdf_filename in pdf_filenames:
            if pdf_filename in outline_data:
                pdf_to_outline_mapping[pdf_filename] = outline_data[pdf_filename]
                matched_count += 1
            else:
                print(f"Warning: No outline found for PDF: {pdf_filename}")
        
        if not pdf_to_outline_mapping:
            print("Error: No PDFs could be mapped to outlines")
            return False
            
        print(f"Successfully mapped {matched_count}/{len(pdf_filenames)} PDFs to outlines")
        
        # Step 3: Extract text from PDFs
        print("\n=== Step 3: Extracting text from PDFs ===")
        pdfs_folder = os.path.join(collection_folder, "PDFs")
        
        if not os.path.exists(pdfs_folder):
            print(f"Error: PDFs folder not found: {pdfs_folder}")
            return False
        
        pdf_sections = process_pdfs_with_outlines(pdf_to_outline_mapping, pdfs_folder)
        
        if not pdf_sections:
            print("Error: No PDF sections extracted")
            return False
        
        total_sections = sum(len(sections) for sections in pdf_sections.values())
        print(f"Extracted {total_sections} sections from {len(pdf_sections)} PDFs")
        
        # Step 4: Perform semantic search
        print("\n=== Step 4: Performing semantic search ===")
        search_results = search_pdf_sections(
            pdf_sections,
            persona,
            job,
            top_k=5
        )
        
        if not search_results:
            print("Warning: No relevant sections found")
            # Still continue to create output file
        else:
            print(f"Found {len(search_results)} relevant sections")
        
        # Step 5: Generate summaries
        print("\n=== Step 5: Generating summaries ===")
        refined_summaries = create_refined_summaries(search_results, max_sentences=3)
        
        # Step 6: Create output JSON
        print("\n=== Step 6: Creating output JSON ===")
        
        # Extract challenge number from input filename
        challenge_num = "1"
        if "challenge" in input_filename.lower():
            import re
            match = re.search(r'challenge(\d+)', input_filename.lower())
            if match:
                challenge_num = match.group(1)
        
        output_filename = f"challenge{challenge_num}_output.json"
        output_path = os.path.join(collection_folder, output_filename)
        
        # Create output data structure
        output_data = {
            "collection": collection_name,
            "persona": persona,
            "job_to_be_done": job,
            "processing_summary": {
                "total_pdfs_requested": len(pdf_filenames),
                "pdfs_successfully_processed": len(pdf_sections),
                "total_sections_extracted": total_sections,
                "relevant_sections_found": len(search_results)
            },
            "top_relevant_sections": refined_summaries
        }
        
        # Save output JSON
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Output saved to: {output_path}")
        print(f"📊 Processing Summary:")
        print(f"   - PDFs processed: {len(pdf_sections)}/{len(pdf_filenames)}")
        print(f"   - Sections extracted: {total_sections}")
        print(f"   - Relevant sections: {len(refined_summaries)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {collection_name}: {e}")
        return False


def run_multi_collection_pipeline(root_dir: str = ".") -> Dict[str, bool]:
    """
    Run the complete pipeline on all Collection folders.
    
    Args:
        root_dir (str): Root directory to search for Collection folders
        
    Returns:
        Dict[str, bool]: Results for each collection (True = success, False = failure)
    """
    print("🚀 STARTING MULTI-COLLECTION PDF PROCESSING PIPELINE")
    print("="*80)
    
    # Find all collection folders
    collection_folders = find_collection_folders(root_dir)
    
    if not collection_folders:
        print("No Collection folders found!")
        return {}
    
    # Process each collection
    results = {}
    successful = 0
    
    for collection_folder in collection_folders:
        collection_name = os.path.basename(collection_folder)
        success = process_single_collection(collection_folder)
        results[collection_name] = success
        
        if success:
            successful += 1
    
    # Print final summary
    print(f"\n{'='*80}")
    print("PIPELINE COMPLETE - FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"✅ Successfully processed: {successful}/{len(collection_folders)} collections")
    
    for collection_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"   {collection_name}: {status}")
    
    if successful == len(collection_folders):
        print(f"\n🎉 All collections processed successfully!")
    elif successful > 0:
        print(f"\n⚠️  {len(collection_folders) - successful} collections had issues")
    else:
        print(f"\n💥 No collections were processed successfully")
    
    return results


def main():
    """
    Main function to run the pipeline.
    """
    # Change to the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)  # Go up one level from /app to root
    
    print(f"Script directory: {script_dir}")
    print(f"Searching for collections in: {parent_dir}")
    
    # Run the pipeline
    results = run_multi_collection_pipeline(parent_dir)
    
    # Exit with appropriate code
    if results and all(results.values()):
        sys.exit(0)  # Success
    else:
        sys.exit(1)  # Some failures


if __name__ == "__main__":
    main()