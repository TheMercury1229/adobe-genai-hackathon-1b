"""
Integrated PDF Text Extraction that uses the existing process_and_map_json.py mapping function.
This processes all PDFs listed in the input JSON, using their outline mappings when available.
Updated version with configurable collection folder path.
"""

import fitz  # PyMuPDF
import json
import os
from typing import List, Dict, Any

# Import your existing mapping function
from process_and_map_json import process_json_with_outlines


def extract_pdf_outline_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """
    Extract the built-in outline/table of contents from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        List[Dict]: List of outline items with text and page numbers
    """
    try:
        pdf_document = fitz.open(pdf_path)
        outline = pdf_document.get_toc()  # Get table of contents
        pdf_document.close()
        
        if not outline:
            return []
        
        # Convert PyMuPDF outline format to our format
        formatted_outline = []
        for item in outline:
            level, title, page_num = item
            formatted_outline.append({
                "level": f"H{level}",
                "text": title.strip(),
                "page": page_num
            })
        
        return formatted_outline
    
    except Exception as e:
        print(f"Error extracting outline from {pdf_path}: {e}")
        return []


def extract_sections_from_pdf_with_outline(pdf_path: str, outline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract text sections from PDF based on outline structure.
    
    Args:
        pdf_path (str): Path to the PDF file
        outline (List[Dict]): Outline structure from JSON or extracted from PDF
        
    Returns:
        List[Dict]: List of sections with title, start_page, end_page, and section_text
    """
    try:
        pdf_document = fitz.open(pdf_path)
        total_pages = len(pdf_document)
        
        if not outline:
            # If no outline, treat entire document as one section
            full_text = ""
            for page_num in range(total_pages):
                try:
                    page = pdf_document[page_num]
                    page_text = page.get_text()
                    full_text += page_text + "\n"
                except Exception as e:
                    print(f"Error extracting text from page {page_num + 1}: {e}")
            
            pdf_document.close()
            return [{
                "section_title": f"Full Document - {os.path.basename(pdf_path)}",
                "start_page": 1,
                "end_page": total_pages,
                "section_text": full_text.strip()
            }]
        
        sections = []
        
        for i, heading in enumerate(outline):
            section_title = heading.get("text", "")
            start_page = heading.get("page", 1)
            
            # Convert to 0-based indexing (PyMuPDF uses 0-based page numbers)
            start_page_idx = start_page - 1
            
            # Determine end page
            if i + 1 < len(outline):  # Not the last heading
                end_page = outline[i + 1].get("page", total_pages + 1) - 1
            else:  # Last heading - go to end of document
                end_page = total_pages
            
            end_page_idx = end_page - 1
            
            # Ensure page indices are within bounds
            start_page_idx = max(0, min(start_page_idx, total_pages - 1))
            end_page_idx = max(start_page_idx, min(end_page_idx, total_pages - 1))
            
            # Extract text from the page range
            section_text = ""
            for page_num in range(start_page_idx, end_page_idx + 1):
                try:
                    page = pdf_document[page_num]
                    page_text = page.get_text()
                    section_text += page_text + "\n"
                except Exception as e:
                    print(f"Error extracting text from page {page_num + 1}: {e}")
            
            # Create section dictionary
            section = {
                "section_title": section_title,
                "start_page": start_page,
                "end_page": end_page,
                "section_text": section_text.strip()
            }
            
            sections.append(section)
            print(f"  Extracted section: '{section_title}' (pages {start_page}-{end_page})")
        
        pdf_document.close()
        return sections
    
    except Exception as e:
        print(f"Error processing PDF {pdf_path}: {e}")
        return []


def process_all_pdfs_with_mapping(collection_folder):
    """
    Process all PDFs using the existing mapping function from process_and_map_json.py
    
    Args:
        collection_folder (str): Path to the collection folder (e.g., "../Collection1")
        
    Returns:
        Dict containing persona, job, and all extracted sections
    """
    pdfs_folder = os.path.join(collection_folder, "PDFs")
    
    print("=" * 60)
    print("INTEGRATED PDF PROCESSING WITH EXISTING MAPPING")
    print("=" * 60)
    
    # Use your existing function to get the mapping
    persona, job, pdf_filenames, pdf_to_outline_mapping = process_json_with_outlines(collection_folder)
    
    if not pdf_filenames:
        print("No PDF filenames found from the mapping function")
        return {}
    
    print(f"\n{'='*60}")
    print("STARTING PDF TEXT EXTRACTION")
    print(f"{'='*60}")
    print(f"Collection folder: {collection_folder}")
    print(f"Persona: {persona}")
    print(f"Job to be done: {job}")
    print(f"PDFs to process: {len(pdf_filenames)}")
    
    all_pdf_sections = {}
    
    for pdf_filename in pdf_filenames:
        pdf_path = os.path.join(pdfs_folder, pdf_filename)
        
        print(f"\n{'-'*50}")
        print(f"Processing: {pdf_filename}")
        print(f"{'-'*50}")
        
        if not os.path.exists(pdf_path):
            print(f"Warning: PDF file not found: {pdf_path}")
            continue
        
        # Check if we have an outline mapping for this PDF
        outline = []
        if pdf_filename in pdf_to_outline_mapping:
            outline_data = pdf_to_outline_mapping[pdf_filename]
            outline = outline_data.get("outline", [])
            if outline:
                print(f"Using JSON outline with {len(outline)} sections")
            else:
                print("JSON outline found but no outline structure")
        
        # If no outline from JSON, try to extract from PDF
        if not outline:
            print("No JSON outline available, trying to extract from PDF...")
            outline = extract_pdf_outline_from_pdf(pdf_path)
            
            if outline:
                print(f"Extracted {len(outline)} outline items from PDF")
            else:
                print("No outline found in PDF, will process as single section")
        
        # Extract sections using the outline
        sections = extract_sections_from_pdf_with_outline(pdf_path, outline)
        all_pdf_sections[pdf_filename] = sections
        
        print(f"Successfully extracted {len(sections)} sections from {pdf_filename}")
        
        # Show section details
        for i, section in enumerate(sections, 1):
            text_length = len(section['section_text'])
            print(f"    {i}. {section['section_title']} -> {text_length:,} characters")
    
    # Compile results
    results = {
        "collection_folder": collection_folder,
        "persona": persona,
        "job_to_be_done": job,
        "pdf_filenames": pdf_filenames,
        "pdf_sections": all_pdf_sections,
        "processing_summary": {
            "total_pdfs": len(pdf_filenames),
            "successfully_processed": len(all_pdf_sections),
            "total_sections": sum(len(sections) for sections in all_pdf_sections.values())
        }
    }
    
    return results


def save_extracted_sections(results: Dict[str, Any], collection_folder: str, output_file: str = "complete_pdf_extraction.json"):
    """
    Save extracted sections and metadata to a JSON file in the collection folder.
    
    Args:
        results (dict): Complete results dictionary
        collection_folder (str): Path to the collection folder
        output_file (str): Output filename for the JSON file
    """
    try:
        output_path = os.path.join(collection_folder, output_file)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nComplete extraction results saved to: {output_path}")
        
        # Also create human-readable summary
        summary_file = output_file.replace('.json', '_summary.txt')
        summary_path = os.path.join(collection_folder, summary_file)
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("COMPLETE PDF EXTRACTION SUMMARY\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Collection folder: {results.get('collection_folder', 'Unknown')}\n")
            f.write(f"Persona: {results['persona']}\n")
            f.write(f"Job to be done: {results['job_to_be_done']}\n\n")
            
            summary = results['processing_summary']
            f.write(f"Processing Summary:\n")
            f.write(f"  - Total PDFs: {summary['total_pdfs']}\n")
            f.write(f"  - Successfully processed: {summary['successfully_processed']}\n")
            f.write(f"  - Total sections extracted: {summary['total_sections']}\n\n")
            
            f.write("Detailed Breakdown:\n")
            f.write("-" * 40 + "\n\n")
            
            for pdf_filename, sections in results['pdf_sections'].items():
                f.write(f"PDF: {pdf_filename}\n")
                f.write(f"Sections: {len(sections)}\n")
                
                total_chars = sum(len(section['section_text']) for section in sections)
                f.write(f"Total characters: {total_chars:,}\n")
                
                for i, section in enumerate(sections, 1):
                    f.write(f"  {i}. {section['section_title']} ")
                    f.write(f"(pages {section['start_page']}-{section['end_page']}) ")
                    f.write(f"- {len(section['section_text']):,} chars\n")
                f.write("\n")
        
        print(f"Human-readable summary saved to: {summary_path}")
        
    except Exception as e:
        print(f"Error saving results: {e}")


def print_final_summary(results: Dict[str, Any]):
    """
    Print a comprehensive summary of the extraction results.
    
    Args:
        results (dict): Complete results dictionary
    """
    print(f"\n{'='*60}")
    print("FINAL EXTRACTION SUMMARY")
    print(f"{'='*60}")
    
    print(f"Collection: {results.get('collection_folder', 'Unknown')}")
    print(f"Persona: {results['persona']}")
    print(f"Task: {results['job_to_be_done']}")
    
    summary = results['processing_summary']
    print(f"\nProcessing Results:")
    print(f"  • PDFs requested: {summary['total_pdfs']}")
    print(f"  • PDFs processed: {summary['successfully_processed']}")
    print(f"  • Total sections: {summary['total_sections']}")
    
    total_characters = 0
    print(f"\nPer-PDF Breakdown:")
    for pdf_filename, sections in results['pdf_sections'].items():
        chars = sum(len(section['section_text']) for section in sections)
        total_characters += chars
        print(f"  • {pdf_filename}: {len(sections)} sections, {chars:,} characters")
    
    print(f"\nTotal extracted text: {total_characters:,} characters")
    if len(results['pdf_sections']) > 0:
        print(f"Average per PDF: {total_characters // len(results['pdf_sections']):,} characters")
    print(f"{'='*60}")


if __name__ == "__main__":
    # Default collection folder for standalone execution
    default_collection = "../Collection1"
    
    print("Starting integrated PDF processing using existing mapping...")
    
    # Process all PDFs using your existing mapping
    results = process_all_pdfs_with_mapping(default_collection)
    
    if results and 'pdf_sections' in results:
        # Print comprehensive summary
        print_final_summary(results)
        
        # Save complete results
        save_extracted_sections(results, default_collection)
        
        print(f"\n🎉 Processing complete!")
        print(f"📄 Check '{default_collection}/complete_pdf_extraction.json' for full data")
        print(f"📋 Check '{default_collection}/complete_pdf_extraction_summary.txt' for readable summary")
        
    else:
        print("❌ No results to save. Check the mapping function and file paths.")