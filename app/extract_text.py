"""
PDF Text Extraction using PyMuPDF based on outline structure.
"""

import fitz  # PyMuPDF
import json
import os
from typing import List, Dict, Any


def extract_sections_from_pdf(pdf_path: str, outline_json_path: str) -> List[Dict[str, Any]]:
    """
    Extract text sections from PDF based on outline structure from JSON file.

    Args:
        pdf_path (str): Path to the PDF file
        outline_json_path (str): Path to the outline JSON file

    Returns:
        List[Dict]: List of sections with title, start_page, end_page, and section_text
    """
    try:
        # Load outline structure from JSON file
        with open(outline_json_path, 'r', encoding='utf-8') as f:
            outline_structure = json.load(f)

        # Open the PDF document
        pdf_document = fitz.open(pdf_path)
        total_pages = len(pdf_document)

        # Get the outline from the structure
        outline = outline_structure.get("outline", [])

        if not outline:
            print(f"Warning: No outline found for {pdf_path}")
            pdf_document.close()
            return []

        sections = []

        for i, heading in enumerate(outline):
            section_title = heading.get("text", "")
            # Default to page 1 if not specified
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
            end_page_idx = max(start_page_idx, min(
                end_page_idx, total_pages - 1))

            # Extract text from the page range
            section_text = ""
            for page_num in range(start_page_idx, end_page_idx + 1):
                try:
                    page = pdf_document[page_num]
                    page_text = page.get_text()
                    section_text += page_text + "\n"
                except Exception as e:
                    print(
                        f"Error extracting text from page {page_num + 1}: {e}")

            # Create section dictionary
            section = {
                "section_title": section_title,
                "start_page": start_page,
                "end_page": end_page,
                "section_text": section_text.strip()
            }

            sections.append(section)

            print(
                f"Extracted section: '{section_title}' (pages {start_page}-{end_page})")

        pdf_document.close()
        return sections

    except FileNotFoundError as e:
        if "outline" in str(e).lower():
            print(f"Error: Outline JSON file not found: {outline_json_path}")
        else:
            print(f"Error: PDF file not found: {pdf_path}")
        return []
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {outline_json_path}")
        return []
    except Exception as e:
        print(
            f"Error processing PDF {pdf_path} with outline {outline_json_path}: {e}")
        return []


def process_pdfs_with_outlines(pdf_to_outline_mapping: Dict[str, Dict], pdfs_folder: str = "pdfs") -> Dict[str, List[Dict]]:
    """
    Process multiple PDFs using their outline mappings.

    Args:
        pdf_to_outline_mapping (dict): Dictionary mapping PDF filenames to their outline structures
        pdfs_folder (str): Folder containing the PDF files

    Returns:
        Dict[str, List[Dict]]: Dictionary mapping PDF filenames to their extracted sections
    """
    pdf_sections = {}

    for pdf_filename, outline_structure in pdf_to_outline_mapping.items():
        pdf_path = os.path.join(pdfs_folder, pdf_filename)

        print(f"\nProcessing PDF: {pdf_filename}")
        print(f"PDF path: {pdf_path}")

        if not os.path.exists(pdf_path):
            print(f"Warning: PDF file not found: {pdf_path}")
            continue

        sections = extract_sections_from_pdf(pdf_path, outline_structure)
        pdf_sections[pdf_filename] = sections

        print(f"Extracted {len(sections)} sections from {pdf_filename}")

    return pdf_sections


def save_extracted_sections(pdf_sections: Dict[str, List[Dict]], output_file: str = "extracted_sections.json"):
    """
    Save extracted sections to a JSON file.

    Args:
        pdf_sections (dict): Dictionary of extracted sections
        output_file (str): Output filename for the JSON file
    """
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(pdf_sections, f, indent=2, ensure_ascii=False)
        print(f"\nExtracted sections saved to: {output_file}")
    except Exception as e:
        print(f"Error saving extracted sections: {e}")


def print_sections_summary(pdf_sections: Dict[str, List[Dict]]):
    """
    Print a summary of extracted sections.

    Args:
        pdf_sections (dict): Dictionary of extracted sections
    """
    print("\n" + "="*60)
    print("EXTRACTED SECTIONS SUMMARY")
    print("="*60)

    for pdf_filename, sections in pdf_sections.items():
        print(f"\nPDF: {pdf_filename}")
        print(f"Total sections: {len(sections)}")

        for i, section in enumerate(sections, 1):
            print(
                f"  {i}. {section['section_title']} (pages {section['start_page']}-{section['end_page']})")
            text_length = len(section['section_text'])
            print(f"     Text length: {text_length} characters")


# Example usage function to integrate with your existing code
def integrate_with_existing_code():
    """
    Example of how to integrate this with your existing process_and_map_json.py code.
    """
    # Import your existing function
    from process_and_map_json import process_json_with_outlines

    # Get the mappings from your existing code
    persona, job, pdf_filenames, pdf_to_outline_mapping = process_json_with_outlines()

    if pdf_filenames:
        # Process PDFs and extract sections using the list of filenames
        pdf_sections = process_pdfs_with_outlines(pdf_filenames)

        # Print summary
        print_sections_summary(pdf_sections)

        # Save to file
        save_extracted_sections(pdf_sections)

        return pdf_sections
    else:
        print("No PDF filenames available")
        return {}


if __name__ == "__main__":
    # Example usage with a single PDF
    example_outline = {
        "filename": "South of France - Cuisine.pdf",
        "title": "A Culinary Journey Through the South of France",
        "outline": [
            {
                "level": "H1",
                "text": "Famous Dishes",
                "page": 3
            },
            {
                "level": "H1",
                "text": "Must-Visit Restaurants",
                "page": 4
            },
            {
                "level": "H1",
                "text": "Wine Regions and Types of Wines",
                "page": 5
            },
            {
                "level": "H1",
                "text": "Culinary Experiences",
                "page": 6
            },
            {
                "level": "H1",
                "text": "Conclusion",
                "page": 8
            }
        ]
    }

    # Extract sections from single PDF
    sections = extract_sections_from_pdf(
        "Collection1/PDFs/South of France - Cuisine.pdf", "Collection1/JSON/South of France - Cuisine.json")

    # Print results
    for section in sections:
        print(f"\nSection: {section['section_title']}")
        print(f"Pages: {section['start_page']}-{section['end_page']}")
        print(f"Text preview: {section['section_text'][:200]}...")
