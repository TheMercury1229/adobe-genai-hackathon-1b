#!/usr/bin/env python3
"""
Collection PDF Processing Script
Processes PDFs from Collection folders based on challenge1b_input.json
Directory structure:
- Collection1/PDFs/        (input PDFs)
- Collection1/JSON/        (output JSON files)
- app/                     (this script and create_csv.py)
- enhanced_*.joblib        (model files in root)
"""

from datetime import datetime
import os
import sys
import json
import pandas as pd
import joblib
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add the app directory to Python path if needed
script_dir = Path(__file__).parent
root_dir = script_dir.parent
sys.path.append(str(script_dir))

try:
    from create_csv import (
        extract_pdf_features,
        filter_significant_elements,
    )
    CSV_MODULE_AVAILABLE = True
except ImportError as e:
    CSV_MODULE_AVAILABLE = False
    print(f"❌ Could not import create_csv.py: {e}")
    print("Make sure create_csv.py is in the app directory")


def load_trained_model():
    """Load the trained model and associated files from root directory"""
    try:
        # Model files are in the root directory (one level up from app)
        model_path = root_dir / 'enhanced_pdf_heading_rf_model.joblib'
        encoder_path = root_dir / 'enhanced_label_encoder.joblib'
        metadata_path = root_dir / 'enhanced_model_metadata.json'

        # Load model components
        model = joblib.load(str(model_path))
        label_encoder = joblib.load(str(encoder_path))

        # Load metadata
        with open(str(metadata_path), 'r') as f:
            metadata = json.load(f)

        print(f"✅ Model loaded successfully from {root_dir}")
        return model, label_encoder, metadata

    except FileNotFoundError as e:
        print(f"❌ Model file not found: {e}")
        print(f"Expected model files in: {root_dir}")
        return None, None, None
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None, None


def preprocess_csv_for_prediction(df, expected_features):
    """Preprocess CSV data to match training format - same as training preprocessing"""

    # Convert numeric columns
    numeric_columns = [
        'page_num', 'block_num', 'line_num', 'font_size', 'position_x', 'position_y',
        'line_width', 'line_height', 'page_width', 'page_height', 'char_count',
        'word_count', 'distance_from_left', 'distance_from_right',
        'distance_from_top', 'distance_from_bottom', 'space_above', 'space_below',
        'font_size_ratio'
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            nan_count = df[col].isnull().sum()
            if nan_count > 0:
                fill_value = df[col].median() if col != 'page_num' else 1
                df[col] = df[col].fillna(fill_value)

    # Create relative coordinates if missing
    if 'page_width' in df.columns and 'page_height' in df.columns:
        if 'relative_x' not in df.columns:
            df['position_x'] = pd.to_numeric(
                df['position_x'], errors='coerce').fillna(0)
            df['page_width'] = pd.to_numeric(
                df['page_width'], errors='coerce').fillna(1)
            df['page_width'] = df['page_width'].replace(0, 1)
            df['relative_x'] = df['position_x'] / df['page_width']
            df['relative_x'] = df['relative_x'].fillna(0)

        if 'relative_y' not in df.columns:
            df['position_y'] = pd.to_numeric(
                df['position_y'], errors='coerce').fillna(0)
            df['page_height'] = pd.to_numeric(
                df['page_height'], errors='coerce').fillna(1)
            df['page_height'] = df['page_height'].replace(0, 1)
            df['relative_y'] = df['position_y'] / df['page_height']
            df['relative_y'] = df['relative_y'].fillna(0)

    # Convert boolean columns
    boolean_columns = [
        'is_bold', 'is_italic', 'is_superscript', 'is_all_caps',
        'is_title_case', 'has_numbers', 'starts_with_number',
        'is_largest_font', 'is_above_avg_font'
    ]

    for col in boolean_columns:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.upper()
            boolean_map = {
                'TRUE': 1, 'FALSE': 0, '1': 1, '0': 0,
                'YES': 1, 'NO': 0, 'T': 1, 'F': 0,
                '1.0': 1, '0.0': 0, 'NAN': 0, 'NONE': 0
            }
            df[col] = df[col].map(boolean_map)
            df[col] = pd.to_numeric(
                df[col], errors='coerce').fillna(0).astype(int)

    # Create composite features
    formatting_features = ['is_bold', 'is_italic',
                           'is_all_caps', 'is_title_case']
    available_formatting = [
        col for col in formatting_features if col in df.columns]

    if available_formatting:
        df['formatting_score'] = df[available_formatting].sum(axis=1)

    if 'is_bold' in df.columns and 'font_size_ratio' in df.columns:
        df['is_bold'] = pd.to_numeric(df['is_bold'], errors='coerce').fillna(0)
        df['font_size_ratio'] = pd.to_numeric(
            df['font_size_ratio'], errors='coerce').fillna(1.0)
        df['font_emphasis'] = df['is_bold'] * df['font_size_ratio']

    # Position-based features
    if 'relative_y' in df.columns and 'relative_x' in df.columns:
        df['relative_y'] = pd.to_numeric(
            df['relative_y'], errors='coerce').fillna(0)
        df['relative_x'] = pd.to_numeric(
            df['relative_x'], errors='coerce').fillna(0)
        df['is_top_third'] = (df['relative_y'] < 0.33).astype(int)
        df['is_left_aligned'] = (df['relative_x'] < 0.1).astype(int)
    else:
        df['is_top_third'] = 0
        df['is_left_aligned'] = 0

    # Text length categories
    if 'char_count' in df.columns:
        df['char_count'] = pd.to_numeric(
            df['char_count'], errors='coerce').fillna(0)
        df['is_short_text'] = (df['char_count'] <= 5).astype(int)
        df['is_medium_text'] = ((df['char_count'] > 5) & (
            df['char_count'] <= 50)).astype(int)
        df['is_long_text'] = (df['char_count'] > 50).astype(int)
    else:
        df['is_short_text'] = 0
        df['is_medium_text'] = 1
        df['is_long_text'] = 0

    # Ensure all expected features exist
    for feature in expected_features:
        if feature not in df.columns:
            print(f"   ⚠️  Missing feature '{feature}', setting to 0")
            df[feature] = 0

    # Fill any remaining missing values
    for col in expected_features:
        if df[col].isnull().sum() > 0:
            if col in boolean_columns + ['formatting_score']:
                df[col] = df[col].fillna(0)
            else:
                df[col] = df[col].fillna(df[col].median())

    print(f"   ✅ Preprocessed shape: {df.shape}")
    return df


def create_outline_json(df, pdf_name):
    """Create JSON outline structure from predictions"""

    # Filter predictions - separate title from other headings
    predictions_df = df[~df['predicted_label'].isin(
        ['body_text', 'body'])].copy()

    # Extract title from model predictions (first occurrence of 'title' prediction)
    title_rows = predictions_df[predictions_df['predicted_label'].str.lower(
    ) == 'title']

    if not title_rows.empty:
        # Use the first predicted title, sorted by page and position
        title_rows_sorted = title_rows.sort_values(['page_num', 'position_y'])
        document_title = title_rows_sorted.iloc[0]['text'].strip()
    else:
        # Fallback: if no title predicted, use filename
        document_title = pdf_name.replace('_', ' ').replace('-', ' ').title()
        print(
            f"   ⚠️  No title predicted by model, using filename: {document_title}")

    # Filter out title from outline (keep only H1, H2, H3, etc.)
    headings_df = predictions_df[~predictions_df['predicted_label'].str.lower().isin([
        'title'])].copy()

    # Sort headings by page number and position
    headings_df = headings_df.sort_values(['page_num', 'position_y'])

    # Create outline structure
    outline = []
    for _, row in headings_df.iterrows():
        heading_item = {
            "level": row['predicted_label'].upper(),  # H1, H2, etc.
            "text": row['text'].strip(),
            "page": int(row['page_num'])
        }
        outline.append(heading_item)

    # Create the final JSON structure
    json_structure = {
        "title": document_title,
        "outline": outline
    }

    return json_structure


def process_single_pdf(pdf_path, json_output_dir, model, label_encoder, expected_features):
    """Process a single PDF through the complete pipeline"""

    pdf_name = Path(pdf_path).stem

    try:
        print(f"   🔧 Extracting features from {pdf_name}...")

        # Step 1: Extract features using create_csv.py functions
        features = extract_pdf_features(pdf_path)

        if not features:
            print(f"   ❌ No features extracted from {pdf_name}")
            return None

        # Step 2: Filter significant elements
        filtered_features, filter_stats = filter_significant_elements(features)

        if not filtered_features:
            print(f"   ⚠️  No significant elements after filtering, using original data")
            filtered_features = features

        # Step 3: Create DataFrame
        df = pd.DataFrame(filtered_features)

        # Step 4: Preprocess for prediction
        print(f"   🔧 Preprocessing for prediction...")
        df_processed = preprocess_csv_for_prediction(
            df.copy(), expected_features)

        # Step 5: Make predictions
        print(f"   🔮 Making predictions...")
        X = df_processed[expected_features].copy()

        predictions = model.predict(X)
        prediction_probs = model.predict_proba(X)
        predicted_labels = label_encoder.inverse_transform(predictions)
        confidence_scores = np.max(prediction_probs, axis=1)

        # Step 6: Create results dataframe
        results_df = df.copy()
        results_df['predicted_label'] = predicted_labels
        results_df['confidence'] = confidence_scores

        # Step 7: Create JSON outline
        json_outline = create_outline_json(results_df, pdf_name)

        # Step 8: Save JSON outline
        json_filename = f"{pdf_name}.json"
        json_path = os.path.join(json_output_dir, json_filename)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_outline, f, indent=4, ensure_ascii=False)

        # Step 9: Print summary
        pred_counts = pd.Series(predicted_labels).value_counts()
        print(f"   📈 Prediction Summary:")
        for label, count in pred_counts.items():
            percentage = (count / len(predicted_labels)) * 100
            print(f"      {label}: {count} ({percentage:.1f}%)")

        print(f"   ✅ JSON saved: {json_path}")

        return {
            'pdf_name': pdf_name,
            'json_path': json_path,
            'total_elements': len(filtered_features),
            'total_headings': len(json_outline['outline']),
            'predictions': dict(pred_counts),
            'mean_confidence': confidence_scores.mean()
        }

    except Exception as e:
        print(f"   ❌ Error processing {pdf_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def find_collection_directories():
    """Find all Collection directories in the root directory"""
    collections = []
    for item in root_dir.iterdir():
        if item.is_dir() and item.name.startswith('Collection'):
            collections.append(item)
    return sorted(collections)


def process_collection(collection_dir):
    """Process a single collection directory"""

    collection_name = collection_dir.name
    print(f"\n{'='*70}")
    print(f"PROCESSING {collection_name}")
    print(f"{'='*70}")

    # Define paths
    pdfs_dir = collection_dir / 'PDFs'
    json_dir = collection_dir / 'JSON'
    input_json_path = collection_dir / 'challenge1b_input.json'

    # Check if required directories and files exist
    if not pdfs_dir.exists():
        print(f"❌ PDFs directory not found: {pdfs_dir}")
        return None

    if not input_json_path.exists():
        print(f"❌ Input JSON file not found: {input_json_path}")
        return None

    # Create JSON output directory if it doesn't exist
    json_dir.mkdir(exist_ok=True)
    print(f"📁 JSON output directory: {json_dir}")

    # Load input JSON file
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        print(f"✅ Loaded input JSON: {len(input_data)} entries")
    except Exception as e:
        print(f"❌ Error loading input JSON: {e}")
        return None

    # Get list of PDF files to process
    pdf_files = []
    if isinstance(input_data, list):
        # If input_data is a list of filenames or objects
        for item in input_data:
            if isinstance(item, str):
                pdf_files.append(item)
            elif isinstance(item, dict) and 'filename' in item:
                pdf_files.append(item['filename'])
            elif isinstance(item, dict) and 'file' in item:
                pdf_files.append(item['file'])
    elif isinstance(input_data, dict):
        # If input_data is a dictionary with file list
        if 'files' in input_data:
            pdf_files = input_data['files']
        elif 'pdfs' in input_data:
            pdf_files = input_data['pdfs']

    if not pdf_files:
        # Fallback: process all PDFs in the directory
        pdf_files = [f.name for f in pdfs_dir.glob('*.pdf')]
        print(
            f"⚠️  No files specified in JSON, processing all PDFs: {len(pdf_files)} files")

    print(f"📄 PDFs to process: {len(pdf_files)}")
    for i, pdf_file in enumerate(pdf_files, 1):
        print(f"   {i}. {pdf_file}")

    return pdf_files, pdfs_dir, json_dir


def main():
    """
    Main entry point for the PDF to JSON conversion process
    Returns structured results for use by sequential processing
    """

    print("="*70)
    print("COLLECTION PDF PROCESSING PIPELINE")
    print("="*70)
    print(f"Root directory: {root_dir}")
    print(f"App directory: {script_dir}")

    # Initialize results structure
    results = {
        "start_time": datetime.now().isoformat(),
        "success": False,
        "collections_processed": 0,
        "collections_failed": 0,
        "total_pdfs_processed": 0,
        "total_pdfs_failed": 0,
        "collection_results": {},
        "errors": []
    }

    try:
        # Check if CSV creation module is available
        if not CSV_MODULE_AVAILABLE:
            error_msg = "Cannot proceed without create_csv.py module"
            print(f"\n❌ {error_msg}")
            print("Please ensure create_csv.py is in the app directory")
            results["errors"].append(error_msg)
            return results

        # Load trained model
        print("\n🤖 Loading trained model...")
        model, label_encoder, metadata = load_trained_model()
        if model is None:
            error_msg = "Cannot proceed without trained model"
            print(f"❌ {error_msg}")
            results["errors"].append(error_msg)
            return results

        expected_features = metadata['feature_columns']
        print(f"✅ Model expects {len(expected_features)} features")

        # Find collection directories
        collections = find_collection_directories()
        if not collections:
            error_msg = f"No Collection directories found in {root_dir}"
            print(f"❌ {error_msg}")
            results["errors"].append(error_msg)
            return results

        print(f"\n📁 Found {len(collections)} collection(s):")
        for collection in collections:
            print(f"   - {collection.name}")

        # Process each collection
        for collection_dir in collections:
            collection_name = collection_dir.name

            try:
                result = process_collection(collection_dir)
                if result is None:
                    results["collections_failed"] += 1
                    results["collection_results"][collection_name] = {
                        "success": False,
                        "error": "Collection processing returned None"
                    }
                    continue

                pdf_files, pdfs_dir, json_dir = result
                collection_results = []
                successful = 0
                failed = 0

                # Process each PDF in the collection
                for i, pdf_file in enumerate(pdf_files, 1):
                    pdf_path = pdfs_dir / pdf_file

                    if not pdf_path.exists():
                        print(
                            f"\n[{i}/{len(pdf_files)}] ❌ PDF not found: {pdf_file}")
                        failed += 1
                        continue

                    print(f"\n[{i}/{len(pdf_files)}] " + "="*50)
                    print(f"Processing: {pdf_file}")

                    pdf_result = process_single_pdf(
                        str(pdf_path),
                        str(json_dir),
                        model,
                        label_encoder,
                        expected_features
                    )

                    if pdf_result:
                        collection_results.append(pdf_result)
                        successful += 1
                        print(f"✅ Successfully processed: {pdf_file}")
                    else:
                        failed += 1
                        print(f"❌ Failed to process: {pdf_file}")

                # Update results for this collection
                results["total_pdfs_processed"] += successful
                results["total_pdfs_failed"] += failed

                if successful > 0:
                    results["collections_processed"] += 1
                    results["collection_results"][collection_name] = {
                        "success": True,
                        "pdfs_processed": successful,
                        "pdfs_failed": failed,
                        "total_pdfs": len(pdf_files),
                        "results": collection_results
                    }
                else:
                    results["collections_failed"] += 1
                    results["collection_results"][collection_name] = {
                        "success": False,
                        "pdfs_processed": 0,
                        "pdfs_failed": failed,
                        "total_pdfs": len(pdf_files),
                        "error": "No PDFs processed successfully"
                    }

                # Collection summary
                print(f"\n{'='*50}")
                print(f"{collection_name} SUMMARY")
                print(f"{'='*50}")
                print(
                    f"✅ Successfully processed: {successful}/{len(pdf_files)} PDFs")

                if collection_results:
                    total_elements = sum(r['total_elements']
                                         for r in collection_results)
                    total_headings = sum(r['total_headings']
                                         for r in collection_results)
                    avg_confidence = sum(
                        r['mean_confidence'] for r in collection_results) / len(collection_results)

                    print(f"📊 Total elements processed: {total_elements}")
                    print(f"📋 Total headings detected: {total_headings}")
                    print(f"🎯 Average confidence: {avg_confidence:.3f}")

            except Exception as e:
                error_msg = f"Error processing collection {collection_name}: {str(e)}"
                print(f"❌ {error_msg}")
                results["errors"].append(error_msg)
                results["collections_failed"] += 1
                results["collection_results"][collection_name] = {
                    "success": False,
                    "error": error_msg
                }

        # Final summary and success determination
        results["end_time"] = datetime.now().isoformat()

        # Consider it successful if at least one collection was processed
        if results["collections_processed"] > 0:
            results["success"] = True

        # Print final summary
        print(f"\n{'='*70}")
        print("FINAL SUMMARY")
        print(f"{'='*70}")

        print(
            f"📁 Collections processed: {results['collections_processed']}/{len(collections)}")
        print(
            f"✅ Total PDFs processed successfully: {results['total_pdfs_processed']}")
        print(f"❌ Total PDFs failed: {results['total_pdfs_failed']}")

        for collection_name, stats in results["collection_results"].items():
            if stats["success"]:
                print(
                    f"   {collection_name}: {stats['pdfs_processed']}/{stats['total_pdfs']} PDFs")
            else:
                print(
                    f"   {collection_name}: FAILED - {stats.get('error', 'Unknown error')}")

        if results["success"]:
            print(f"\n🎉 PDF to JSON conversion completed successfully!")
        else:
            print(f"\n⚠️  PDF to JSON conversion completed with issues")

    except Exception as e:
        error_msg = f"Fatal error in main processing: {str(e)}"
        print(f"❌ {error_msg}")
        results["errors"].append(error_msg)
        results["end_time"] = datetime.now().isoformat()

        # Print traceback for debugging
        import traceback
        traceback.print_exc()

    return results


if __name__ == "__main__":
    try:
        # Run the main function and get results
        results = main()

        # Exit with appropriate code
        if results["success"]:
            print(f"\n✅ Process completed successfully!")
            sys.exit(0)
        else:
            print(f"\n❌ Process completed with errors")
            if results["errors"]:
                print(f"Errors: {'; '.join(results['errors'])}")
            sys.exit(1)

    except Exception as e:
        print(f"\n💥 Fatal error: {e}")
        sys.exit(2)
