"""
Pipeline for Processing Collection Folders
This pipeline automatically discovers and processes all Collection folders (Collection1, Collection2, etc.).

Pipeline Flow:
1. Discovers all ../Collection* folders
2. For each Collection, runs the three-step process in sequence:
   - Step 1: process_and_map_json.py (mapping PDF filenames to outlines)
   - Step 2: extract_text.py (PDF text extraction using mapping from step 1)
   - Step 3: embedder.py (semantic search using extraction from step 2)

Output: Each Collection folder will contain the complete processing results
"""

import os
import sys
import glob
import logging
import traceback
from datetime import datetime
from typing import List, Dict, Any, Tuple
import json

# Add the current directory to Python path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the three processing modules
try:
    from process_and_map_json import process_json_with_outlines, print_mapping_summary
    from extract_text import process_all_pdfs_with_mapping, save_extracted_sections, print_final_summary
    from embedder import process_collection_with_semantic_search
    MODULES_IMPORTED = True
    print("✅ All required modules imported successfully")
except ImportError as e:
    print(f"❌ Error importing modules: {e}")
    print("Make sure all three files (process_and_map_json.py, extract_text.py, embedder.py) are in the same directory as pipeline.py")
    MODULES_IMPORTED = False


class CollectionPipeline:
    """
    Pipeline class for processing Collection folders in sequence
    """
    
    def __init__(self, base_path: str = ".."):
        self.base_path = base_path
        self.setup_logging()
        self.processing_summary = {
            "start_time": datetime.now().isoformat(),
            "collections_found": 0,
            "collections_processed": 0,
            "collections_failed": 0,
            "processing_results": []
        }
    
    def setup_logging(self):
        """Setup comprehensive logging for the pipeline"""
        # Create logs directory if it doesn't exist
        log_dir = "pipeline_logs"
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging with both file and console output
        log_filename = os.path.join(log_dir, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Pipeline logging initialized. Log file: {log_filename}")
    
    def discover_collection_folders(self) -> List[str]:
        """
        Discover all Collection folders in the base path (Collection1, Collection2, etc.)
        
        Returns:
            List[str]: List of Collection folder paths
        """
        collection_pattern = os.path.join(self.base_path, "Collection*")
        collection_folders = glob.glob(collection_pattern)
        
        # Filter to only include directories
        collection_folders = [folder for folder in collection_folders if os.path.isdir(folder)]
        collection_folders.sort()  # Sort for consistent processing order
        
        self.logger.info(f"Discovered {len(collection_folders)} Collection folders:")
        for folder in collection_folders:
            self.logger.info(f"  📁 {folder}")
        
        self.processing_summary["collections_found"] = len(collection_folders)
        return collection_folders
    
    def validate_collection_structure(self, collection_folder: str) -> Tuple[bool, List[str]]:
        """
        Validate that a Collection folder has the required structure
        
        Args:
            collection_folder (str): Path to the Collection folder
            
        Returns:
            Tuple[bool, List[str]]: (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for required files and folders
        required_items = {
            "challenge1b_input.json": "file",
            "PDFs": "folder",
            "JSON": "folder"
        }
        
        for item_name, item_type in required_items.items():
            item_path = os.path.join(collection_folder, item_name)
            
            if item_type == "file" and not os.path.isfile(item_path):
                issues.append(f"Missing required file: {item_name}")
            elif item_type == "folder" and not os.path.isdir(item_path):
                issues.append(f"Missing required folder: {item_name}")
        
        # Check if PDFs folder has PDF files
        pdfs_folder = os.path.join(collection_folder, "PDFs")
        if os.path.isdir(pdfs_folder):
            pdf_files = glob.glob(os.path.join(pdfs_folder, "*.pdf"))
            if not pdf_files:
                issues.append("PDFs folder exists but contains no PDF files")
            else:
                self.logger.info(f"  Found {len(pdf_files)} PDF files")
        
        # Check if JSON folder has JSON files
        json_folder = os.path.join(collection_folder, "JSON")
        if os.path.isdir(json_folder):
            json_files = glob.glob(os.path.join(json_folder, "*.json"))
            if not json_files:
                issues.append("JSON folder exists but contains no JSON files")
            else:
                self.logger.info(f"  Found {len(json_files)} JSON files")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def process_single_collection(self, collection_folder: str) -> Dict[str, Any]:
        """
        Process a single Collection folder through the entire pipeline
        
        Args:
            collection_folder (str): Path to the Collection folder
            
        Returns:
            Dict[str, Any]: Processing results and status
        """
        collection_name = os.path.basename(collection_folder)
        
        result = {
            "collection": collection_name,
            "collection_path": collection_folder,
            "start_time": datetime.now().isoformat(),
            "success": False,
            "steps_completed": [],
            "steps_failed": [],
            "output_files": [],
            "errors": [],
            "step_outputs": {}  # Store outputs from each step for chaining
        }
        
        self.logger.info(f"{'='*80}")
        self.logger.info(f"PROCESSING COLLECTION: {collection_name}")
        self.logger.info(f"Path: {collection_folder}")
        self.logger.info(f"{'='*80}")
        
        try:
            # Step 0: Validate collection structure
            self.logger.info("Step 0: Validating collection structure...")
            is_valid, issues = self.validate_collection_structure(collection_folder)
            
            if not is_valid:
                error_msg = f"Collection structure validation failed: {'; '.join(issues)}"
                self.logger.error(error_msg)
                result["errors"].append(error_msg)
                result["steps_failed"].append("validation")
                return result
            
            self.logger.info("✅ Collection structure validation passed")
            result["steps_completed"].append("validation")
            
            # Step 1: Process and map JSON (process_and_map_json.py)
            self.logger.info("\nStep 1: Processing and mapping JSON...")
            self.logger.info("-" * 50)
            try:
                persona, job, pdf_filenames, pdf_to_outline_mapping = process_json_with_outlines(collection_folder)
                
                if not pdf_filenames:
                    raise Exception("No PDF filenames found in challenge1b_input.json")
                
                if not pdf_to_outline_mapping:
                    self.logger.warning("No outline mappings found, but continuing with PDF filenames")
                
                # Store step 1 outputs for use in step 2
                result["step_outputs"]["step1"] = {
                    "persona": persona,
                    "job": job,
                    "pdf_filenames": pdf_filenames,
                    "pdf_to_outline_mapping": pdf_to_outline_mapping
                }
                
                self.logger.info(f"✅ Step 1 completed successfully")
                self.logger.info(f"  • Persona: {persona}")
                self.logger.info(f"  • Job: {job}")
                self.logger.info(f"  • PDF files: {len(pdf_filenames)}")
                self.logger.info(f"  • Outline mappings: {len(pdf_to_outline_mapping)}")
                result["steps_completed"].append("json_mapping")
                
                # Print mapping summary if available
                if pdf_to_outline_mapping:
                    print_mapping_summary(pdf_to_outline_mapping)
                
            except Exception as e:
                error_msg = f"Step 1 (JSON mapping) failed: {str(e)}"
                self.logger.error(error_msg)
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                result["errors"].append(error_msg)
                result["steps_failed"].append("json_mapping")
                return result
            
            # Step 2: Extract text from PDFs (extract_text.py)
            self.logger.info("\nStep 2: Extracting text from PDFs...")
            self.logger.info("-" * 50)
            try:
                # Use the mapping from step 1 - the extract_text module will use process_json_with_outlines internally
                pdf_extraction_results = process_all_pdfs_with_mapping(collection_folder)
                
                if not pdf_extraction_results or 'pdf_sections' not in pdf_extraction_results:
                    raise Exception("PDF text extraction failed or returned no sections")
                
                # Store step 2 outputs for use in step 3
                result["step_outputs"]["step2"] = pdf_extraction_results
                
                # Save the extraction results to the collection folder
                save_extracted_sections(pdf_extraction_results, collection_folder)
                
                # Print summary
                print_final_summary(pdf_extraction_results)
                
                sections_count = sum(len(sections) for sections in pdf_extraction_results.get('pdf_sections', {}).values())
                self.logger.info(f"✅ Step 2 completed successfully")
                self.logger.info(f"  • PDFs processed: {len(pdf_extraction_results.get('pdf_sections', {}))}")
                self.logger.info(f"  • Total sections extracted: {sections_count}")
                
                result["steps_completed"].append("text_extraction")
                result["output_files"].extend([
                    "complete_pdf_extraction.json",
                    "complete_pdf_extraction_summary.txt"
                ])
                
            except Exception as e:
                error_msg = f"Step 2 (text extraction) failed: {str(e)}"
                self.logger.error(error_msg)
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                result["errors"].append(error_msg)
                result["steps_failed"].append("text_extraction")
                return result
            
            # Step 3: Semantic search and embeddings (embedder.py)
            self.logger.info("\nStep 3: Performing semantic search and creating embeddings...")
            self.logger.info("-" * 50)
            try:
                # Use the extraction results from step 2 - the embedder will process the same collection
                semantic_results = process_collection_with_semantic_search(collection_folder, top_k=5)
                
                if not semantic_results:
                    raise Exception("Semantic search failed or returned no results")
                
                # Store step 3 outputs
                result["step_outputs"]["step3"] = semantic_results
                
                extracted_sections = semantic_results.get('extracted_sections', [])
                self.logger.info(f"✅ Step 3 completed successfully")
                self.logger.info(f"  • Top relevant sections found: {len(extracted_sections)}")
                
                result["steps_completed"].append("semantic_search")
                result["output_files"].extend([
                    "semantic_search_results.json",
                    "semantic_search_results_summary.txt"
                ])
                
            except Exception as e:
                error_msg = f"Step 3 (semantic search) failed: {str(e)}"
                self.logger.error(error_msg)
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                result["errors"].append(error_msg)
                result["steps_failed"].append("semantic_search")
                return result
            
            # If we get here, all steps completed successfully
            result["success"] = True
            result["end_time"] = datetime.now().isoformat()
            
            self.logger.info(f"\n🎉 {collection_name} processed successfully!")
            self.logger.info(f"📁 All output files saved to: {collection_folder}")
            self.logger.info(f"📄 Output files: {', '.join(result['output_files'])}")
            
        except Exception as e:
            error_msg = f"Unexpected error processing {collection_name}: {str(e)}"
            self.logger.error(error_msg)
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            result["errors"].append(error_msg)
        
        return result
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete pipeline for all discovered Collection folders
        
        Returns:
            Dict[str, Any]: Complete processing summary
        """
        if not MODULES_IMPORTED:
            self.logger.error("Cannot run pipeline: Required modules not imported")
            return {"error": "Module import failed"}
        
        self.logger.info("🚀 STARTING COLLECTION PROCESSING PIPELINE")
        self.logger.info(f"Base path: {os.path.abspath(self.base_path)}")
        
        # Discover all Collection folders
        collection_folders = self.discover_collection_folders()
        
        if not collection_folders:
            self.logger.warning("No Collection folders found. Exiting.")
            self.logger.info("Expected folder structure: ../Collection1, ../Collection2, etc.")
            return self.processing_summary
        
        # Process each Collection folder
        for collection_folder in collection_folders:
            try:
                collection_result = self.process_single_collection(collection_folder)
                self.processing_summary["processing_results"].append(collection_result)
                
                if collection_result["success"]:
                    self.processing_summary["collections_processed"] += 1
                else:
                    self.processing_summary["collections_failed"] += 1
                    
            except Exception as e:
                error_msg = f"Failed to process {collection_folder}: {str(e)}"
                self.logger.error(error_msg)
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                self.processing_summary["collections_failed"] += 1
                
                # Add failed result to summary
                failed_result = {
                    "collection": os.path.basename(collection_folder),
                    "collection_path": collection_folder,
                    "success": False,
                    "errors": [error_msg],
                    "steps_completed": [],
                    "steps_failed": ["initialization"]
                }
                self.processing_summary["processing_results"].append(failed_result)
        
        # Finalize summary
        self.processing_summary["end_time"] = datetime.now().isoformat()
        
        # Print final summary
        self.print_final_pipeline_summary()
        
        # Save comprehensive summary
        self.save_pipeline_summary()
        
        return self.processing_summary
    
    def print_final_pipeline_summary(self):
        """Print a comprehensive summary of the entire pipeline execution"""
        print(f"\n{'='*100}")
        print("COLLECTION PROCESSING PIPELINE SUMMARY")
        print(f"{'='*100}")
        
        summary = self.processing_summary
        
        print(f"📊 Overall Statistics:")
        print(f"  • Collections found: {summary['collections_found']}")
        print(f"  • Collections processed successfully: {summary['collections_processed']}")
        print(f"  • Collections failed: {summary['collections_failed']}")
        
        if summary['collections_found'] > 0:
            success_rate = (summary['collections_processed'] / summary['collections_found']) * 100
            print(f"  • Success rate: {success_rate:.1f}%")
        
        print(f"\n📋 Detailed Results:")
        successful_collections = [r for r in summary['processing_results'] if r['success']]
        failed_collections = [r for r in summary['processing_results'] if not r['success']]
        
        if successful_collections:
            print(f"\n✅ Successfully Processed ({len(successful_collections)}):")
            for result in successful_collections:
                print(f"  • {result['collection']}")
                print(f"    ✓ Steps: {' → '.join(result['steps_completed'])}")
                print(f"    📄 Files: {len(result.get('output_files', []))} output files created")
        
        if failed_collections:
            print(f"\n❌ Failed to Process ({len(failed_collections)}):")
            for result in failed_collections:
                print(f"  • {result['collection']}")
                if result.get('steps_completed'):
                    print(f"    ✓ Completed: {', '.join(result['steps_completed'])}")
                if result.get('steps_failed'):
                    print(f"    ✗ Failed at: {', '.join(result['steps_failed'])}")
                if result.get('errors'):
                    print(f"    💥 Error: {result['errors'][0]}")
        
        # Processing time
        if 'start_time' in summary and 'end_time' in summary:
            start_time = datetime.fromisoformat(summary['start_time'])
            end_time = datetime.fromisoformat(summary['end_time'])
            duration = end_time - start_time
            print(f"\n⏱️  Total processing time: {duration}")
        
        print(f"\n📁 Output locations:")
        print(f"  • Detailed logs: pipeline_logs/")
        print(f"  • Summary files: pipeline_summary.json & pipeline_summary.txt")
        print(f"  • Collection results: Each Collection folder contains output files")
        print(f"{'='*100}")
    
    def save_pipeline_summary(self):
        """Save comprehensive pipeline summary to JSON file"""
        try:
            summary_file = "pipeline_summary.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(self.processing_summary, f, indent=4, ensure_ascii=False)
            
            self.logger.info(f"Pipeline summary saved to: {summary_file}")
            
            # Also save a human-readable summary
            readable_summary_file = "pipeline_summary.txt"
            with open(readable_summary_file, 'w', encoding='utf-8') as f:
                f.write("COLLECTION PROCESSING PIPELINE SUMMARY\n")
                f.write("=" * 80 + "\n\n")
                
                summary = self.processing_summary
                f.write(f"Processing Statistics:\n")
                f.write(f"  Collections found: {summary['collections_found']}\n")
                f.write(f"  Collections processed: {summary['collections_processed']}\n")
                f.write(f"  Collections failed: {summary['collections_failed']}\n")
                
                if summary['collections_found'] > 0:
                    success_rate = (summary['collections_processed'] / summary['collections_found']) * 100
                    f.write(f"  Success rate: {success_rate:.1f}%\n")
                
                f.write(f"\n")
                
                if 'start_time' in summary and 'end_time' in summary:
                    f.write(f"Start time: {summary['start_time']}\n")
                    f.write(f"End time: {summary['end_time']}\n\n")
                
                f.write("Detailed Results:\n")
                f.write("-" * 40 + "\n\n")
                
                for result in summary['processing_results']:
                    f.write(f"{result['collection']}: ")
                    f.write("SUCCESS\n" if result['success'] else "FAILED\n")
                    
                    if result.get('steps_completed'):
                        f.write(f"  ✓ Completed: {' → '.join(result['steps_completed'])}\n")
                    
                    if result.get('steps_failed'):
                        f.write(f"  ✗ Failed: {', '.join(result['steps_failed'])}\n")
                    
                    if result.get('output_files'):
                        f.write(f"  📄 Output files: {', '.join(result['output_files'])}\n")
                    
                    if result.get('errors'):
                        f.write(f"  💥 Errors: {'; '.join(result['errors'])}\n")
                    
                    f.write("\n")
            
            self.logger.info(f"Readable summary saved to: {readable_summary_file}")
            
        except Exception as e:
            self.logger.error(f"Error saving pipeline summary: {e}")


def main():
    """
    Main execution function for the collection processing pipeline
    """
    print("🚀 Collection Processing Pipeline")
    print("Processing all Collection folders (Collection1, Collection2, etc.)...")
    print("=" * 80)
    
    try:
        # Initialize and run the pipeline
        pipeline = CollectionPipeline(base_path="..")
        results = pipeline.run_complete_pipeline()
        
        if results.get('collections_processed', 0) > 0:
            print(f"\n🎉 Pipeline completed successfully!")
            print(f"✅ Processed {results['collections_processed']} collections")
            if results.get('collections_failed', 0) > 0:
                print(f"⚠️  {results['collections_failed']} collections failed - check logs for details")
        else:
            print(f"\n⚠️  Pipeline completed but no collections were processed successfully")
            if results.get('collections_found', 0) == 0:
                print("💡 Make sure you have Collection folders (Collection1, Collection2, etc.) in the parent directory")
            else:
                print("💡 Check the logs and error messages above for troubleshooting")
        
        return results
        
    except KeyboardInterrupt:
        print(f"\n⏹️  Pipeline interrupted by user")
        return {"error": "Pipeline interrupted"}
    
    except Exception as e:
        print(f"\n❌ Pipeline execution failed: {e}")
        logging.error(f"Pipeline execution error: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")
        return {"error": str(e)}


if __name__ == "__main__":
    # Run the main pipeline
    main()