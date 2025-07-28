"""
Integrated semantic search functionality that works with the existing PDF processing pipeline.
Uses process_and_map_json.py and extract_text.py to process all PDFs in specified collection folder.
Output saved to the same collection folder in the specified JSON format.
Updated version with configurable collection folder path and custom output filenames.
"""

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
from typing import List, Dict, Any, Tuple, Optional
import logging
from datetime import datetime

# Import your existing functions
from process_and_map_json import process_json_with_outlines
from extract_text import process_all_pdfs_with_mapping, save_extracted_sections


class IntegratedSemanticSearch:
    """
    Integrated semantic search that works with existing PDF processing pipeline
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", collection_folder: str = "../Collection1"):
        self.model_name = model_name
        self.model = None
        self.collection_folder = collection_folder
        self.setup_logging()

    def setup_logging(self):
        """Setup logging for semantic search operations"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def load_model(self):
        """Load the sentence transformer model with fallback options"""
        model_options = [
            self.model_name,
            "all-MiniLM-L6-v2",
            "paraphrase-MiniLM-L6-v2",
            "all-mpnet-base-v2"
        ]

        for model_name in model_options:
            try:
                self.logger.info(f"Attempting to load model: {model_name}")
                self.model = SentenceTransformer(model_name)
                self.model_name = model_name  # Update to successful model
                self.logger.info(f"Model loaded successfully: {model_name}")
                return
            except Exception as e:
                self.logger.warning(f"Failed to load {model_name}: {e}")
                continue

        # If all models fail, raise the last error
        raise Exception("Failed to load any sentence transformer model")

    def process_all_pdfs_and_extract(self) -> Dict[str, Any]:
        """
        Use existing pipeline to process all PDFs and extract sections
        """
        self.logger.info("Starting PDF processing using existing pipeline...")

        try:
            # Use your existing function to process all PDFs
            results = process_all_pdfs_with_mapping(self.collection_folder)

            if not results or 'pdf_sections' not in results:
                self.logger.error(
                    "No PDF sections extracted from existing pipeline")
                return {}

            self.logger.info(
                f"Successfully processed {len(results['pdf_sections'])} PDFs")
            return results

        except Exception as e:
            self.logger.error(f"Error in PDF processing pipeline: {e}")
            return {}

    def prepare_sections_for_search(self, pdf_extraction_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Prepare sections from the existing PDF extraction format for semantic search.
        Handles the structure from your extract_text.py output.

        Args:
            pdf_extraction_results: Complete results from process_all_pdfs_with_mapping()

        Returns:
            List[Dict]: Flattened sections ready for semantic search
        """
        all_sections = []
        pdf_sections = pdf_extraction_results.get('pdf_sections', {})

        self.logger.info(
            f"Preparing {len(pdf_sections)} PDFs for semantic search...")

        for filename, sections_list in pdf_sections.items():
            self.logger.info(f"Processing sections from: {filename}")

            for section in sections_list:
                # Create section data for semantic search
                section_data = {
                    'filename': filename,
                    'section_title': section.get('section_title', 'Untitled'),
                    'start_page': section.get('start_page', 0),
                    'end_page': section.get('end_page', 0),
                    'section_text': section.get('section_text', '').strip(),
                    'character_count': len(section.get('section_text', '').strip()),
                    'word_count': len(section.get('section_text', '').split())
                }

                # Only include sections with meaningful content
                if section_data['section_text'] and len(section_data['section_text']) > 50:
                    all_sections.append(section_data)
                else:
                    self.logger.warning(
                        f"Skipping section with insufficient content: {section_data['section_title']}")

        self.logger.info(
            f"Prepared {len(all_sections)} sections for semantic search")
        return all_sections

    def find_most_relevant_sections(
        self,
        sections_data: List[Dict[str, Any]],
        persona: str,
        job_to_be_done: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find the most relevant PDF sections based on persona and job using semantic similarity.

        Args:
            sections_data: List of prepared section dictionaries
            persona: The persona string (e.g., "Travel Planner")
            job_to_be_done: The job description (e.g., "Plan a trip for 10 friends")
            top_k: Number of top results to return

        Returns:
            List[Dict]: Top k most relevant sections with similarity scores
        """
        if not sections_data:
            self.logger.warning("No sections data provided for search")
            return []

        if not persona.strip() or not job_to_be_done.strip():
            self.logger.warning("Empty persona or job_to_be_done provided")
            return []

        try:
            # Ensure model is loaded
            if self.model is None:
                self.load_model()

            # Create comprehensive query string
            query = f"{persona.strip()}: {job_to_be_done.strip()}"
            self.logger.info(f"Searching with query: '{query}'")

            # Extract section texts and titles for better matching
            search_texts = []
            valid_sections = []

            for section in sections_data:
                section_text = section.get('section_text', '').strip()
                section_title = section.get('section_title', '').strip()

                # Combine title and text for better semantic matching
                combined_text = f"{section_title} {section_text}"

                if combined_text.strip():
                    search_texts.append(combined_text)
                    valid_sections.append(section)

            if not search_texts:
                self.logger.warning("No valid sections with text found")
                return []

            self.logger.info(
                f"Computing embeddings for {len(search_texts)} sections...")

            # Generate embeddings with error handling
            try:
                section_embeddings = self.model.encode(
                    search_texts,
                    convert_to_tensor=False,
                    show_progress_bar=True,
                    batch_size=16  # Reduced batch size for stability
                )

                query_embedding = self.model.encode(
                    [query], convert_to_tensor=False)

            except Exception as e:
                self.logger.error(f"Error generating embeddings: {e}")
                # Fallback: return sections based on keyword matching
                return self._keyword_fallback_search(valid_sections, query, top_k)

            # Compute cosine similarities
            similarities = cosine_similarity(
                query_embedding, section_embeddings)[0]

            # Create results with similarity scores and additional metadata
            results = []
            for i, section in enumerate(valid_sections):
                result = {
                    'filename': section.get('filename', 'Unknown'),
                    'section_title': section.get('section_title', 'Untitled'),
                    'start_page': section.get('start_page', 0),
                    'end_page': section.get('end_page', 0),
                    'similarity_score': float(similarities[i]),
                    'section_text': section.get('section_text', ''),
                    'character_count': section.get('character_count', 0),
                    'word_count': section.get('word_count', 0),
                    'text_preview': section.get('section_text', '')[:300] + "..." if len(section.get('section_text', '')) > 300 else section.get('section_text', '')
                }
                results.append(result)

            # Sort by similarity score (descending) and return top k
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            top_results = results[:top_k]

            self.logger.info(f"Found {len(top_results)} top relevant sections")
            return top_results

        except Exception as e:
            self.logger.error(f"Error in semantic search: {e}")
            # Fallback to keyword search
            self.logger.info("Falling back to keyword-based search...")
            return self._keyword_fallback_search(sections_data, f"{persona} {job_to_be_done}", top_k)

    def _keyword_fallback_search(self, sections_data: List[Dict[str, Any]], query: str, top_k: int) -> List[Dict[str, Any]]:
        """
        Fallback keyword-based search when semantic search fails
        """
        try:
            query_words = set(query.lower().split())
            results = []

            for section in sections_data:
                section_text = section.get('section_text', '').lower()
                section_title = section.get('section_title', '').lower()
                combined_text = f"{section_title} {section_text}"

                # Count keyword matches
                matches = sum(
                    1 for word in query_words if word in combined_text)
                keyword_score = matches / \
                    len(query_words) if query_words else 0

                result = {
                    'filename': section.get('filename', 'Unknown'),
                    'section_title': section.get('section_title', 'Untitled'),
                    'start_page': section.get('start_page', 0),
                    'end_page': section.get('end_page', 0),
                    'similarity_score': keyword_score,  # Using keyword score as similarity
                    'section_text': section.get('section_text', ''),
                    'character_count': section.get('character_count', 0),
                    'word_count': section.get('word_count', 0),
                    'text_preview': section.get('section_text', '')[:300] + "..." if len(section.get('section_text', '')) > 300 else section.get('section_text', ''),
                    'search_method': 'keyword_fallback'
                }
                results.append(result)

            # Sort by keyword score and return top k
            results.sort(key=lambda x: x['similarity_score'], reverse=True)
            top_results = results[:top_k]

            self.logger.info(
                f"Keyword fallback found {len(top_results)} results")
            return top_results

        except Exception as e:
            self.logger.error(f"Error in keyword fallback search: {e}")
            return []

    def format_output_like_sample(
        self,
        pdf_results: Dict[str, Any],
        search_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Format the output to match the sample JSON structure provided
        """
        # Extract PDF filenames from the results
        pdf_sections = pdf_results.get('pdf_sections', {})
        input_documents = list(pdf_sections.keys())

        # Get persona and job info
        persona = pdf_results.get('persona', 'Travel Planner')
        job_to_be_done = pdf_results.get(
            'job_to_be_done', 'Plan a trip of 4 days for a group of 10 college friends.')
        collection_folder = pdf_results.get('collection_folder', 'Unknown')

        # Create metadata section
        metadata = {
            "collection_folder": collection_folder,
            "input_documents": input_documents,
            "persona": persona,
            "job_to_be_done": job_to_be_done,
            "processing_timestamp": datetime.now().isoformat()
        }

        # Create extracted_sections from top search results
        extracted_sections = []
        for i, result in enumerate(search_results, 1):
            section_info = {
                "document": result['filename'],
                "section_title": result['section_title'],
                "importance_rank": i,
                "page_number": result.get('start_page', 1),
                "similarity_score": result.get('similarity_score', 0.0)
            }
            extracted_sections.append(section_info)

        # Create subsection_analysis with refined text from search results
        subsection_analysis = []
        for result in search_results:
            # Use the full section text as refined_text
            refined_text = result['section_text']

            # If text is too long, truncate but keep it meaningful
            if len(refined_text) > 2000:
                # Find a good breaking point (end of sentence)
                truncate_at = refined_text.find('. ', 1800)
                if truncate_at != -1:
                    refined_text = refined_text[:truncate_at + 1]
                else:
                    refined_text = refined_text[:2000] + "..."

            analysis_entry = {
                "document": result['filename'],
                "refined_text": refined_text,
                "page_number": result.get('start_page', 1),
                "similarity_score": result.get('similarity_score', 0.0)
            }
            subsection_analysis.append(analysis_entry)

        # Combine into the final structure
        formatted_output = {
            "metadata": metadata,
            "extracted_sections": extracted_sections,
            "subsection_analysis": subsection_analysis
        }

        return formatted_output

    def complete_pipeline_search(
        self,
        top_k: int = 5,
        save_results: bool = True,
        output_filename: str = "output.json"  # Changed default filename
    ) -> Dict[str, Any]:
        """
        Complete pipeline: process PDFs, extract text, and perform semantic search
        Output saved to collection folder in the specified format

        Args:
            top_k: Number of top results to return
            save_results: Whether to save results to file
            output_filename: Output filename for results (will be saved in collection folder)

        Returns:
            Dict containing formatted results
        """
        self.logger.info("="*80)
        self.logger.info("STARTING COMPLETE SEMANTIC SEARCH PIPELINE")
        self.logger.info("="*80)

        # Step 1: Process all PDFs using existing pipeline
        pdf_results = self.process_all_pdfs_and_extract()

        if not pdf_results:
            self.logger.error("Failed to process PDFs")
            return {}

        # Extract persona and job from the results
        persona = pdf_results.get('persona', 'Travel Planner')
        job_to_be_done = pdf_results.get(
            'job_to_be_done', 'Plan a trip of 4 days for a group of 10 college friends.')
        collection_folder = pdf_results.get(
            'collection_folder', self.collection_folder)

        self.logger.info(f"Collection folder: {collection_folder}")
        self.logger.info(f"Persona: {persona}")
        self.logger.info(f"Job to be done: {job_to_be_done}")

        # Step 2: Prepare sections for semantic search
        sections_data = self.prepare_sections_for_search(pdf_results)

        if not sections_data:
            self.logger.error("No sections prepared for search")
            return {}

        # Step 3: Perform semantic search
        self.logger.info(
            f"Performing semantic search for top {top_k} results...")
        search_results = self.find_most_relevant_sections(
            sections_data, persona, job_to_be_done, top_k
        )

        if not search_results:
            self.logger.error("No search results found")
            return {}

        # Step 4: Format output like the sample JSON
        formatted_results = self.format_output_like_sample(
            pdf_results, search_results)

        # Step 5: Save results to collection folder if requested
        if save_results:
            self.save_formatted_results(formatted_results, output_filename)

        # Step 6: Print summary
        self.print_formatted_summary(formatted_results, output_filename)

        return formatted_results

    def save_formatted_results(self, results: Dict[str, Any], output_filename: str):
        """Save formatted results to collection folder"""
        try:
            # Get current working directory
            current_dir = os.getcwd()
            self.logger.info(f"Current working directory: {current_dir}")

            # Create full path to collection folder
            collection_path = os.path.abspath(self.collection_folder)
            self.logger.info(f"Collection folder path: {collection_path}")

            # Ensure collection folder exists
            os.makedirs(collection_path, exist_ok=True)
            self.logger.info(
                f"Collection folder created/verified: {collection_path}")

            # Full path for output file
            output_path = os.path.join(collection_path, output_filename)
            self.logger.info(f"Attempting to save to: {output_path}")

            # Save formatted results
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=4, ensure_ascii=False)

            # Verify file was created
            if os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                self.logger.info(
                    f"✅ File saved successfully: {output_path} ({file_size} bytes)")
                print(f"✅ JSON file saved: {output_path}")
            else:
                self.logger.error(f"❌ File was not created: {output_path}")
                print(f"❌ Failed to create file: {output_path}")

            # Also save a summary text file
            summary_filename = output_filename.replace('.json', '_summary.txt')
            summary_path = os.path.join(collection_path, summary_filename)
            self.save_text_summary(results, summary_path)

        except Exception as e:
            self.logger.error(f"Error saving formatted results: {e}")
            print(f"❌ Error saving results: {e}")
            # Try saving to current directory as fallback
            try:
                fallback_path = output_filename
                with open(fallback_path, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)
                print(
                    f"💡 Fallback: Saved to current directory: {fallback_path}")
            except Exception as fallback_error:
                print(f"❌ Fallback also failed: {fallback_error}")

    def save_text_summary(self, results: Dict[str, Any], summary_path: str):
        """Save human-readable summary"""
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("SEMANTIC SEARCH RESULTS SUMMARY\n")
                f.write("="*80 + "\n\n")

                metadata = results.get('metadata', {})
                f.write(f"Processing Information:\n")
                f.write(
                    f"  • Collection folder: {metadata.get('collection_folder', 'N/A')}\n")
                f.write(f"  • Persona: {metadata.get('persona', 'N/A')}\n")
                f.write(f"  • Job: {metadata.get('job_to_be_done', 'N/A')}\n")
                f.write(
                    f"  • Documents processed: {len(metadata.get('input_documents', []))}\n")
                f.write(
                    f"  • Processing time: {metadata.get('processing_timestamp', 'N/A')}\n\n")

                f.write("Input Documents:\n")
                for doc in metadata.get('input_documents', []):
                    f.write(f"  • {doc}\n")
                f.write("\n")

                f.write("Top Relevant Sections (by importance):\n")
                f.write("-"*80 + "\n\n")

                for section in results.get('extracted_sections', []):
                    f.write(
                        f"{section['importance_rank']}. {section['section_title']}\n")
                    f.write(f"   Document: {section['document']}\n")
                    f.write(f"   Page: {section['page_number']}\n")
                    f.write(
                        f"   Similarity Score: {section.get('similarity_score', 0.0):.4f}\n\n")

                f.write("Detailed Content Analysis:\n")
                f.write("-"*80 + "\n\n")

                for i, analysis in enumerate(results.get('subsection_analysis', []), 1):
                    f.write(f"{i}. Document: {analysis['document']}\n")
                    f.write(f"   Page: {analysis['page_number']}\n")
                    f.write(
                        f"   Similarity Score: {analysis.get('similarity_score', 0.0):.4f}\n")
                    f.write(
                        f"   Content Preview: {analysis['refined_text'][:200]}...\n")
                    f.write("-"*60 + "\n\n")

            self.logger.info(f"Text summary saved to: {summary_path}")

        except Exception as e:
            self.logger.error(f"Error saving text summary: {e}")

    def print_formatted_summary(self, results: Dict[str, Any], output_filename: str = "output.json"):
        """Print comprehensive summary of formatted results"""
        print("\n" + "="*80)
        print("SEMANTIC SEARCH PIPELINE COMPLETED")
        print("="*80)

        metadata = results.get('metadata', {})
        print(f"\nProcessing Summary:")
        print(f"  • Collection: {metadata.get('collection_folder', 'N/A')}")
        print(f"  • Persona: {metadata.get('persona', 'N/A')}")
        print(f"  • Job: {metadata.get('job_to_be_done', 'N/A')}")
        print(f"  • Documents: {len(metadata.get('input_documents', []))}")
        print(f"  • Timestamp: {metadata.get('processing_timestamp', 'N/A')}")

        extracted_sections = results.get('extracted_sections', [])
        print(f"\nTop {len(extracted_sections)} Most Relevant Sections:")
        print("-"*60)

        for section in extracted_sections:
            print(f"{section['importance_rank']}. {section['section_title']}")
            print(
                f"   📁 {section['document']} (Page {section['page_number']}) - Score: {section.get('similarity_score', 0.0):.4f}")

        print(
            f"\n📊 Results saved to {self.collection_folder}/{output_filename}")
        print(f"📋 Check the JSON file for complete structured data")
        print(f"📝 Check the summary file for readable overview")
        print("\n" + "="*80)


def process_collection_with_semantic_search(
    collection_folder: str,
    top_k: int = 5,
    output_filename: str = "output.json"  # Added output_filename parameter
) -> Dict[str, Any]:
    """
    Process a single collection folder with semantic search

    Args:
        collection_folder (str): Path to the collection folder (e.g., "../Collection1")
        top_k (int): Number of top results to return
        output_filename (str): Name of the output JSON file

    Returns:
        Dict containing formatted results
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING COLLECTION: {collection_folder}")
    print(f"OUTPUT FILE: {output_filename}")
    print(f"{'='*80}")

    try:
        # Initialize semantic search system for this collection
        search_system = IntegratedSemanticSearch(
            model_name="paraphrase-MiniLM-L6-v2",
            collection_folder=collection_folder
        )

        # Run complete pipeline with formatted output
        results = search_system.complete_pipeline_search(
            top_k=top_k,
            save_results=True,
            output_filename=output_filename  # Pass the custom filename
        )

        if results and results.get('extracted_sections'):
            print(
                f"\n🎉 Collection {collection_folder} processed successfully!")
            return results
        else:
            print(
                f"\n⚠️  Collection {collection_folder} processed but no results found")
            return {}

    except Exception as e:
        print(f"\n❌ Error processing collection {collection_folder}: {e}")
        logging.error(f"Collection processing error: {e}")
        return {}


def main():
    """
    Main execution function for the integrated semantic search pipeline
    """
    print("Starting Integrated Semantic Search Pipeline...")
    print("Processing PDFs from Collection folder and saving results there.")

    # Default collection folder for standalone execution
    default_collection = "../Collection1"

    # You can choose which output filename to use:
    # Option 1: Simple output.json
    output_filename = "output.json"

    # Option 2: Challenge-specific output.json (uncomment to use)
    # output_filename = "challenge1b_output.json"

    try:
        results = process_collection_with_semantic_search(
            collection_folder=default_collection,
            output_filename=output_filename
        )

        if results:
            print(
                f"\n📁 Results saved to {default_collection}/{output_filename}")
            summary_filename = output_filename.replace('.json', '_summary.txt')
            print(
                f"📋 Summary saved to {default_collection}/{summary_filename}")
        else:
            print("\n💡 Try the following:")
            print("  • Check your internet connection")
            print("  • Run: pip install sentence-transformers scikit-learn")
            print(
                f"  • Ensure {default_collection} folder exists with PDF files")
            print("  • Verify all imported modules are available")

    except Exception as e:
        print(f"\n❌ Error in pipeline execution: {e}")
        logging.error(f"Pipeline error: {e}")


# Additional convenience functions for different challenges
def process_challenge1b(collection_folder: str = "../Collection1", top_k: int = 5):
    """
    Convenience function to process Challenge 1B with specific output filename
    """
    return process_collection_with_semantic_search(
        collection_folder=collection_folder,
        top_k=top_k,
        output_filename="challenge1b_output.json"
    )


def process_with_output_json(collection_folder: str = "../Collection1", top_k: int = 5):
    """
    Convenience function to process with standard output.json filename
    """
    return process_collection_with_semantic_search(
        collection_folder=collection_folder,
        top_k=top_k,
        output_filename="challenge1b_output.json"
    )


if __name__ == "__main__":
    main()
