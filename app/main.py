#!/usr/bin/env python3
"""
Sequential Processing Script
Runs create_json processing first, then pipeline.py processing in sequence.

This script:
1. First runs the PDF to JSON conversion (from create_json script)
2. Then runs the complete collection processing pipeline

Usage:
    python run_sequential_processing.py
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
import traceback
import json

# Add the current directory to Python path
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))


def setup_logging():
    """Setup logging for the sequential processing"""
    log_dir = "sequential_logs"
    os.makedirs(log_dir, exist_ok=True)

    log_filename = os.path.join(
        log_dir, f"sequential_processing_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger = logging.getLogger(__name__)
    logger.info(
        f"Sequential processing logging initialized. Log file: {log_filename}")
    return logger


def import_and_run_create_json():
    """Import and run the main function from the create_json script"""
    print("="*80)
    print("STEP 1: RUNNING PDF TO JSON CONVERSION")
    print("="*80)

    try:
        # Import the main function from the create_json script
        from create_json import main as create_json_main

        print("✅ Successfully imported create_json main function")

        # Run the create_json main function
        result = create_json_main()

        print("✅ PDF to JSON conversion completed")

        # Return structured result
        return {
            "success": True,
            "result": result,
            "step": "create_json"
        }

    except ImportError as e:
        error_msg = f"Could not import create_json main function: {e}"
        print(f"❌ {error_msg}")
        print("Make sure the create_json script is available and has a main() function")
        return {
            "success": False,
            "error": error_msg,
            "step": "create_json"
        }
    except Exception as e:
        error_msg = f"Error running create_json processing: {e}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return {
            "success": False,
            "error": error_msg,
            "step": "create_json"
        }


def import_and_run_pipeline():
    """Import and run the main function from the pipeline script"""
    print("\n" + "="*80)
    print("STEP 2: RUNNING COLLECTION PROCESSING PIPELINE")
    print("="*80)

    try:
        # Import the main function from the pipeline script
        from pipeline import main as pipeline_main

        print("✅ Successfully imported pipeline main function")

        # Run the pipeline main function
        result = pipeline_main()

        print("✅ Collection processing pipeline completed")

        # Return structured result
        return {
            "success": True,
            "result": result,
            "step": "pipeline"
        }

    except ImportError as e:
        error_msg = f"Could not import pipeline main function: {e}"
        print(f"❌ {error_msg}")
        print("Make sure the pipeline script is available and has a main() function")
        return {
            "success": False,
            "error": error_msg,
            "step": "pipeline"
        }
    except Exception as e:
        error_msg = f"Error running pipeline processing: {e}"
        print(f"❌ {error_msg}")
        traceback.print_exc()
        return {
            "success": False,
            "error": error_msg,
            "step": "pipeline"
        }


def validate_environment():
    """Validate that the environment is set up correctly"""
    print("🔍 VALIDATING ENVIRONMENT")
    print("-" * 50)

    validation_results = {
        "create_json_available": False,
        "pipeline_available": False,
        "collection_folders_found": False,
        "issues": []
    }

    # Check if create_json module is available
    try:
        import create_json
        if hasattr(create_json, 'main'):
            validation_results["create_json_available"] = True
            print("✅ create_json.py with main() function found")
        else:
            validation_results["issues"].append(
                "create_json.py found but no main() function")
            print("⚠️  create_json.py found but no main() function")
    except ImportError:
        validation_results["issues"].append(
            "create_json.py not found or not importable")
        print("❌ create_json.py not found or not importable")

    # Check if pipeline module is available
    try:
        import pipeline
        if hasattr(pipeline, 'main'):
            validation_results["pipeline_available"] = True
            print("✅ pipeline.py with main() function found")
        else:
            validation_results["issues"].append(
                "pipeline.py found but no main() function")
            print("⚠️  pipeline.py found but no main() function")
    except ImportError:
        validation_results["issues"].append(
            "pipeline.py not found or not importable")
        print("❌ pipeline.py not found or not importable")

    # Check for Collection folders
    collection_folders = list(Path("..").glob("Collection*"))
    if collection_folders:
        validation_results["collection_folders_found"] = True
        print(f"✅ Found {len(collection_folders)} Collection folders:")
        for folder in sorted(collection_folders):
            print(f"   📁 {folder.name}")
    else:
        validation_results["issues"].append(
            "No Collection folders found in parent directory")
        print("❌ No Collection folders found in parent directory")

    # Summary
    all_valid = (validation_results["create_json_available"] and
                 validation_results["pipeline_available"] and
                 validation_results["collection_folders_found"])

    if all_valid:
        print("✅ Environment validation passed")
    else:
        print("⚠️  Environment validation found issues:")
        for issue in validation_results["issues"]:
            print(f"   • {issue}")

    return validation_results


def main():
    """Main function to run both processes sequentially"""
    logger = setup_logging()

    print("🚀 SEQUENTIAL PROCESSING PIPELINE")
    print("Running PDF to JSON conversion followed by collection processing")
    print("="*80)

    start_time = datetime.now()
    logger.info("Starting sequential processing pipeline")

    # Summary tracking
    summary = {
        "start_time": start_time.isoformat(),
        "step1_success": False,
        "step2_success": False,
        "step1_result": None,
        "step2_result": None,
        "errors": [],
        "validation_results": None,
        "continue_after_step1_failure": False
    }

    try:
        # Step 0: Validate environment
        logger.info("Step 0: Environment validation")
        validation_results = validate_environment()
        summary["validation_results"] = validation_results

        if not validation_results["create_json_available"] and not validation_results["pipeline_available"]:
            error_msg = "Critical validation failure: Neither step can be executed"
            logger.error(error_msg)
            summary["errors"].append(error_msg)
            return summary

        # Step 1: Run create_json processing
        if validation_results["create_json_available"]:
            logger.info("Starting Step 1: PDF to JSON conversion")
            step1_result = import_and_run_create_json()

            if step1_result["success"]:
                summary["step1_success"] = True
                summary["step1_result"] = step1_result["result"]
                logger.info("✅ Step 1 completed successfully")
            else:
                logger.error(
                    "❌ Step 1 failed - PDF to JSON conversion unsuccessful")
                summary["errors"].append(
                    f"Step 1 failed: {step1_result.get('error', 'Unknown error')}")

                # Ask user if they want to continue despite step 1 failure
                if validation_results["pipeline_available"]:
                    user_input = input(
                        "\n⚠️  Step 1 failed. Continue with Step 2 anyway? (y/n): ").lower().strip()
                    if user_input in ['y', 'yes']:
                        summary["continue_after_step1_failure"] = True
                        logger.info(
                            "User chose to continue despite Step 1 failure")
                    else:
                        print("Stopping execution as requested.")
                        logger.info("User chose to stop after Step 1 failure")
                        return summary
                else:
                    logger.error(
                        "Cannot continue: Pipeline module not available")
                    return summary
        else:
            logger.warning("Skipping Step 1: create_json module not available")
            summary["errors"].append(
                "Step 1 skipped: create_json module not available")

        # Step 2: Run pipeline processing
        if validation_results["pipeline_available"] and (summary["step1_success"] or summary["continue_after_step1_failure"]):
            logger.info("Starting Step 2: Collection processing pipeline")
            step2_result = import_and_run_pipeline()

            if step2_result["success"]:
                summary["step2_success"] = True
                summary["step2_result"] = step2_result["result"]
                logger.info("✅ Step 2 completed successfully")
            else:
                logger.error(
                    "❌ Step 2 failed - Collection processing pipeline unsuccessful")
                summary["errors"].append(
                    f"Step 2 failed: {step2_result.get('error', 'Unknown error')}")
        elif not validation_results["pipeline_available"]:
            logger.warning("Skipping Step 2: pipeline module not available")
            summary["errors"].append(
                "Step 2 skipped: pipeline module not available")
        else:
            logger.info(
                "Skipping Step 2: Step 1 failed and user chose not to continue")

    except KeyboardInterrupt:
        print(f"\n⏹️  Sequential processing interrupted by user")
        logger.info("Sequential processing interrupted by user")
        summary["errors"].append("Process interrupted by user")
        return summary

    except Exception as e:
        error_msg = f"Unexpected error in sequential processing: {e}"
        print(f"\n❌ {error_msg}")
        logger.error(error_msg)
        logger.error(f"Traceback: {traceback.format_exc()}")
        summary["errors"].append(error_msg)

    finally:
        # Calculate total time and finalize summary
        end_time = datetime.now()
        total_duration = end_time - start_time
        summary["end_time"] = end_time.isoformat()
        summary["total_duration"] = str(total_duration)

        # Print final summary
        print_final_summary(summary, logger)

        # Save summary to file
        save_summary(summary)

    return summary


def print_final_summary(summary, logger):
    """Print a comprehensive summary of the sequential processing"""
    print(f"\n{'='*80}")
    print("SEQUENTIAL PROCESSING SUMMARY")
    print(f"{'='*80}")

    # Overall status
    step1_status = "✅ SUCCESS" if summary["step1_success"] else "❌ FAILED"
    step2_status = "✅ SUCCESS" if summary["step2_success"] else "❌ FAILED"

    print(f"📊 Processing Results:")
    print(f"   Step 1 (PDF to JSON): {step1_status}")
    print(f"   Step 2 (Pipeline): {step2_status}")

    # Overall assessment
    if summary["step1_success"] and summary["step2_success"]:
        print(f"🎉 Both steps completed successfully!")
        overall_status = "SUCCESS"
    elif summary["step1_success"] or summary["step2_success"]:
        print(f"⚠️  Partial success - one step completed successfully")
        overall_status = "PARTIAL_SUCCESS"
    else:
        print(f"❌ Both steps failed")
        overall_status = "FAILED"

    # Timing information
    if "total_duration" in summary:
        print(f"⏱️  Total processing time: {summary['total_duration']}")

    # Error information
    if summary["errors"]:
        print(f"\n❌ Errors encountered:")
        for i, error in enumerate(summary["errors"], 1):
            print(f"   {i}. {error}")

    # Additional details from individual steps
    if summary["step1_success"] and summary["step1_result"]:
        print(f"\n📄 Step 1 Details: PDF to JSON conversion completed")

    if summary["step2_success"] and summary["step2_result"]:
        step2_result = summary["step2_result"]
        if isinstance(step2_result, dict):
            collections_processed = step2_result.get(
                "collections_processed", "unknown")
            collections_failed = step2_result.get(
                "collections_failed", "unknown")
            print(
                f"📁 Step 2 Details: {collections_processed} collections processed, {collections_failed} failed")

    # Output files information
    print(f"\n📁 Output files:")
    print(f"   • Sequential processing logs: sequential_logs/")
    print(f"   • Sequential summary: sequential_processing_summary.json")
    if summary["step2_success"]:
        print(f"   • Pipeline outputs: Check individual Collection folders")
        print(f"   • Pipeline logs: pipeline_logs/")

    print(f"{'='*80}")

    logger.info(
        f"Sequential processing completed with status: {overall_status}")


def save_summary(summary):
    """Save the processing summary to a JSON file"""
    try:
        summary_file = "sequential_processing_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=4, ensure_ascii=False)

        print(f"📄 Sequential processing summary saved to: {summary_file}")

    except Exception as e:
        print(f"⚠️  Could not save summary file: {e}")


if __name__ == "__main__":
    try:
        # Run the sequential processing
        results = main()

        # Exit with appropriate code based on results
        if results["step1_success"] and results["step2_success"]:
            print(f"\n🎉 Sequential processing completed successfully!")
            sys.exit(0)
        elif results["step1_success"] or results["step2_success"]:
            print(f"\n⚠️  Sequential processing completed with partial success")
            sys.exit(1)
        else:
            print(f"\n❌ Sequential processing failed")
            sys.exit(2)

    except Exception as e:
        print(f"\n💥 Fatal error in sequential processing: {e}")
        sys.exit(3)
