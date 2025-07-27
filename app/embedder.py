"""
Semantic search functionality for PDF sections using sentence transformers.
Extends the existing PDF text extraction with semantic similarity matching.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Tuple


def find_most_relevant_sections(
    sections_data: List[Dict[str, Any]],
    persona: str,
    job_to_be_done: str,
    model_name: str = "all-MiniLM-L6-v2",
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Find the most relevant PDF sections based on persona and job using semantic similarity.

    Args:
        sections_data (List[Dict]): List of section dictionaries containing:
            - section_title: str
            - start_page: int
            - section_text: str
            - filename: str (should be added to existing sections)
        persona (str): The persona string (e.g., "Travel Planner")
        job_to_be_done (str): The job description (e.g., "Plan a trip for 10 friends")
        model_name (str): Sentence transformer model name
        top_k (int): Number of top results to return

    Returns:
        List[Dict]: Top k most relevant sections with similarity scores, sorted by relevance
    """

    if not sections_data:
        return []

    if not persona.strip() or not job_to_be_done.strip():
        return []

    try:
        # Load the sentence transformer model
        model = SentenceTransformer(model_name)

        # Create query string by concatenating persona and job
        query = f"{persona.strip()} {job_to_be_done.strip()}"

        # Extract section texts for embedding
        section_texts = []
        valid_sections = []

        for section in sections_data:
            section_text = section.get('section_text', '').strip()
            if section_text:  # Only include sections with text
                section_texts.append(section_text)
                valid_sections.append(section)

        if not section_texts:
            return []

        # Generate embeddings for all section texts
        section_embeddings = model.encode(
            section_texts, convert_to_tensor=False, show_progress_bar=False)

        # Generate embedding for the query
        query_embedding = model.encode([query], convert_to_tensor=False)

        # Compute cosine similarities
        similarities = cosine_similarity(
            query_embedding, section_embeddings)[0]

        # Create results with similarity scores
        results = []
        for i, section in enumerate(valid_sections):
            result = {
                'filename': section.get('filename', 'Unknown'),
                'section_title': section.get('section_title', 'Untitled'),
                'start_page': section.get('start_page', 0),
                'similarity_score': float(similarities[i]),
                'section_text': section.get('section_text', '')
            }
            results.append(result)

        # Sort by similarity score (descending) and return top k
        results.sort(key=lambda x: x['similarity_score'], reverse=True)
        top_results = results[:top_k]

        return top_results

    except Exception as e:
        print(f"Error in semantic search: {e}")
        return []


def prepare_sections_for_search(pdf_sections: Dict[str, List[Dict]]) -> List[Dict[str, Any]]:
    """
    Prepare sections from the existing PDF extraction format for semantic search.
    Adds filename to each section and flattens the structure.

    Args:
        pdf_sections (Dict[str, List[Dict]]): Output from process_pdfs_with_outlines()

    Returns:
        List[Dict]: Flattened sections with filename added
    """
    all_sections = []

    for filename, sections in pdf_sections.items():
        for section in sections:
            # Add filename to each section
            section_with_filename = section.copy()
            section_with_filename['filename'] = filename
            all_sections.append(section_with_filename)

    return all_sections


def search_pdf_sections(
    pdf_sections: Dict[str, List[Dict]],
    persona: str,
    job_to_be_done: str,
    model_name: str = "all-MiniLM-L6-v2",
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Complete pipeline: prepare sections and perform semantic search.

    Args:
        pdf_sections (Dict[str, List[Dict]]): Output from process_pdfs_with_outlines()
        persona (str): The persona string
        job_to_be_done (str): The job description
        model_name (str): Sentence transformer model name
        top_k (int): Number of top results to return

    Returns:
        List[Dict]: Top k most relevant sections
    """
    # Prepare sections for search
    sections_data = prepare_sections_for_search(pdf_sections)

    # Perform semantic search
    return find_most_relevant_sections(
        sections_data, persona, job_to_be_done, model_name, top_k
    )


def print_search_results(results: List[Dict[str, Any]]):
    """
    Pretty print the search results.

    Args:
        results (List[Dict]): Results from semantic search
    """
    if not results:
        print("No results found.")
        return

    print("\n" + "="*80)
    print("SEMANTIC SEARCH RESULTS")
    print("="*80)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['section_title']}")
        print(f"   File: {result['filename']}")
        print(f"   Page: {result['start_page']}")
        print(f"   Similarity Score: {result['similarity_score']:.4f}")
        print(f"   Text Preview: {result['section_text'][:200]}...")
        print("-" * 60)


# Integration example with existing code
def integrated_example():
    """
    Example of how to integrate semantic search with existing PDF extraction.
    """
    # This would use your existing extraction function
    # from your_existing_module import process_pdfs_with_outlines

    # Example data structure (replace with actual extraction results)
    example_pdf_sections = {
        "South of France - Cuisine.pdf": [
            {
                "section_title": "Famous Dishes",
                "start_page": 3,
                "end_page": 3,
                "section_text": "The South of France is renowned for its Mediterranean cuisine featuring fresh seafood, olive oil, and aromatic herbs. Bouillabaisse is the most famous dish from Marseille..."
            },
            {
                "section_title": "Must-Visit Restaurants",
                "start_page": 4,
                "end_page": 4,
                "section_text": "For an authentic culinary experience, visit these renowned establishments. La Petite Maison in Nice offers exceptional Mediterranean dishes..."
            }
        ]
    }

    # Perform semantic search
    persona = "Travel Planner"
    job_to_be_done = "Plan a culinary trip for food enthusiasts"

    results = search_pdf_sections(
        example_pdf_sections,
        persona,
        job_to_be_done,
        top_k=5
    )

    # Print results
    print_search_results(results)

    return results


if __name__ == "__main__":
    # Run the integrated example
    integrated_example()
