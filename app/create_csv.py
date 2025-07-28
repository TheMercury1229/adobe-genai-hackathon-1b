import fitz  # PyMuPDF
import pandas as pd
from collections import Counter


def extract_pdf_features(pdf_path):
    """
    Extract detailed styling features from each line of a PDF document

    Args:
        pdf_path (str): Path to the PDF file

    Returns:
        list: List of dictionaries containing features for each line
    """
    doc = fitz.open(pdf_path)
    all_elements = []

    for page_num in range(doc.page_count):
        page = doc[page_num]
        page_width = page.rect.width
        page_height = page.rect.height

        print(f"--- PAGE {page_num + 1} ---")
        print(f"Page dimensions: {page_width:.1f} x {page_height:.1f}")

        # Get text blocks with formatting information
        blocks = page.get_text("dict")

        page_elements = []

        for block_num, block in enumerate(blocks["blocks"]):
            if "lines" in block:  # Text block (not image)
                for line_num, line in enumerate(block["lines"]):
                    if not line["spans"]:  # Skip empty lines
                        continue

                    # Initialize line features
                    line_features = {
                        'page_num': page_num + 1,
                        'block_num': block_num,
                        'line_num': line_num,
                        'text': '',
                        'font_size': 0,
                        'font_name': '',
                        'is_bold': False,
                        'is_italic': False,
                        'is_superscript': False,
                        'position_x': 0,
                        'position_y': 0,
                        'line_width': 0,
                        'line_height': 0,
                        'page_width': page_width,
                        'page_height': page_height,
                        'char_count': 0,
                        'word_count': 0
                    }

                    # Extract text and get dominant formatting
                    line_text_parts = []
                    max_font_size = 0
                    dominant_span = None

                    for span in line["spans"]:
                        span_text = span["text"].strip()
                        if span_text:  # Only process non-empty spans
                            line_text_parts.append(span_text)

                            # Find the span with largest font size (dominant formatting)
                            if span["size"] > max_font_size:
                                max_font_size = span["size"]
                                dominant_span = span

                    # Skip empty lines
                    if not line_text_parts or not dominant_span:
                        continue

                    # Combine all text in the line
                    line_features['text'] = ' '.join(line_text_parts)

                    # Extract dominant formatting features
                    line_features['font_size'] = dominant_span['size']
                    line_features['font_name'] = dominant_span['font']

                    # Check font flags for bold/italic
                    flags = dominant_span['flags']
                    line_features['is_bold'] = bool(
                        flags & 2**4) or 'Bold' in dominant_span['font']
                    line_features['is_italic'] = bool(
                        flags & 2**1) or 'Italic' in dominant_span['font']
                    line_features['is_superscript'] = bool(flags & 2**0)

                    # Position and dimensions
                    bbox = line['bbox']
                    line_features['position_x'] = bbox[0]
                    line_features['position_y'] = bbox[1]
                    line_features['line_width'] = bbox[2] - bbox[0]
                    line_features['line_height'] = bbox[3] - bbox[1]

                    # Text statistics
                    line_features['char_count'] = len(line_features['text'])
                    line_features['word_count'] = len(
                        line_features['text'].split())

                    page_elements.append(line_features)

        # Calculate derived features for the page
        page_elements = calculate_derived_features(
            page_elements, page_width, page_height)
        all_elements.extend(page_elements)

        # Print page summary
        print(f"Lines extracted: {len(page_elements)}\n")

    doc.close()

    # Calculate document-level features
    all_elements = calculate_document_features(all_elements)

    return all_elements


def calculate_derived_features(elements, page_width, page_height):
    """Calculate additional features based on position and spacing"""

    for i, element in enumerate(elements):
        # Horizontal positioning
        center_x = page_width / 2
        if abs(element['position_x'] - center_x) < 30:  # Within 30 points of center
            element['horizontal_position'] = 'center'
        elif element['position_x'] < page_width * 0.3:
            element['horizontal_position'] = 'left'
        elif element['position_x'] > page_width * 0.7:
            element['horizontal_position'] = 'right'
        else:
            element['horizontal_position'] = 'middle'

        # Relative position on page
        element['relative_x'] = element['position_x'] / page_width
        element['relative_y'] = element['position_y'] / page_height

        # Distance from page edges
        element['distance_from_left'] = element['position_x']
        element['distance_from_right'] = page_width - \
            (element['position_x'] + element['line_width'])
        element['distance_from_top'] = element['position_y']
        element['distance_from_bottom'] = page_height - element['position_y']

        # Text characteristics
        element['is_all_caps'] = element['text'].isupper()
        element['is_title_case'] = element['text'].istitle()
        element['has_numbers'] = any(char.isdigit()
                                     for char in element['text'])
        element['starts_with_number'] = element['text'][0].isdigit(
        ) if element['text'] else False

        # Calculate spacing with previous and next lines
        if i > 0:
            prev_element = elements[i-1]
            element['space_above'] = element['position_y'] - \
                (prev_element['position_y'] + prev_element['line_height'])
        else:
            element['space_above'] = element['distance_from_top']

        if i < len(elements) - 1:
            next_element = elements[i+1]
            element['space_below'] = next_element['position_y'] - \
                (element['position_y'] + element['line_height'])
        else:
            element['space_below'] = element['distance_from_bottom']

    return elements


def calculate_document_features(all_elements):
    """Calculate document-level relative features"""

    if not all_elements:
        return all_elements

    # Find font size statistics
    font_sizes = [elem['font_size'] for elem in all_elements]
    max_font_size = max(font_sizes)
    avg_font_size = sum(font_sizes) / len(font_sizes)

    for element in all_elements:
        # Relative font size features
        element['font_size_ratio'] = element['font_size'] / max_font_size
        element['is_largest_font'] = element['font_size'] == max_font_size
        element['is_above_avg_font'] = element['font_size'] > avg_font_size

    return all_elements


def detect_repeated_elements(elements, header_threshold=0.1, footer_threshold=0.9):
    """
    Detect headers and footers based on repetition across maximum pages.
    Also considers position thresholds for additional filtering.

    For documents with 1-2 pages, relaxes repetition criteria to avoid
    incorrectly flagging legitimate content as repeated elements.

    Args:
        elements (list): List of extracted elements (already font-filtered)
        header_threshold (float): Y position threshold for headers (0-1, relative to page)
        footer_threshold (float): Y position threshold for footers (0-1, relative to page)

    Returns:
        dict: Contains sets of repeated element indices and statistics
    """
    if not elements:
        return {'repeated_indices': set(), 'stats': {}}

    # Get total number of pages
    total_pages = max(elem['page_num'] for elem in elements)

    # Group elements by their text content (case-insensitive, stripped)
    text_groups = {}
    for i, elem in enumerate(elements):
        text_key = elem['text'].strip().lower()

        # Skip very short text (less than 3 characters) unless it's likely a page number
        if len(text_key) < 3 and not (text_key.isdigit() or any(p in text_key for p in ['page', 'p.', 'pg'])):
            continue

        if text_key not in text_groups:
            text_groups[text_key] = []
        text_groups[text_key].append((i, elem))

    repeated_indices = set()
    repetition_stats = []

    # Find text that appears on many pages (likely headers/footers)
    for text_key, occurrences in text_groups.items():
        # Get unique pages where this text appears
        pages_with_text = set(elem['page_num'] for _, elem in occurrences)
        num_pages_with_text = len(pages_with_text)

        # Calculate repetition ratio
        repetition_ratio = num_pages_with_text / total_pages

        # Determine if this should be considered repeated
        is_repeated = False
        position_based = False

        # For very short documents (1-2 pages), be more restrictive
        if total_pages <= 2:
            # Only flag as repeated if it's clearly in header/footer position
            # AND appears on all pages
            if num_pages_with_text == total_pages:
                avg_rel_y = sum(elem['relative_y']
                                for _, elem in occurrences) / len(occurrences)
                if avg_rel_y <= header_threshold or avg_rel_y >= footer_threshold:
                    is_repeated = True
                    position_based = True
        else:
            # For longer documents, use the original logic
            if repetition_ratio >= 0.5:  # Appears on 50%+ of pages
                is_repeated = True
            elif num_pages_with_text >= 2:  # Appears on multiple pages
                # Check if it's in header/footer position
                avg_rel_y = sum(elem['relative_y']
                                for _, elem in occurrences) / len(occurrences)
                if avg_rel_y <= header_threshold or avg_rel_y >= footer_threshold:
                    is_repeated = True
                    position_based = True

        if is_repeated:
            # Add all occurrences of this repeated text
            for idx, elem in occurrences:
                repeated_indices.add(idx)

            repetition_stats.append({
                'text': text_key[:50] + ('...' if len(text_key) > 50 else ''),
                'occurrences': len(occurrences),
                'pages': num_pages_with_text,
                'repetition_ratio': repetition_ratio,
                'position_based': position_based,
                'avg_position_y': sum(elem['relative_y'] for _, elem in occurrences) / len(occurrences)
            })

    # Sort stats by repetition ratio (most repeated first)
    repetition_stats.sort(key=lambda x: x['repetition_ratio'], reverse=True)

    return {
        'repeated_indices': repeated_indices,
        'stats': {
            'total_pages': total_pages,
            'repeated_patterns': repetition_stats,
            'total_repeated_elements': len(repeated_indices)
        }
    }


def filter_significant_elements(elements):
    """
    Two-stage filtering process with special handling for bold text and numbered items:
    1. First filter by font ratio to remove body text (but preserve bold/numbered content)
    2. Then remove repeated elements (headers/footers) from remaining content

    Args:
        elements (list): List of extracted elements

    Returns:
        tuple: (filtered_elements, filter_stats)
    """
    if not elements:
        return elements, {}

    original_count = len(elements)

    # ===== STAGE 1: FONT RATIO FILTERING WITH BOLD/NUMBERED PRESERVATION =====

    # Extract font size ratios
    font_ratios = [elem['font_size_ratio'] for elem in elements]

    # Round ratios to avoid floating point precision issues
    rounded_ratios = [round(ratio, 3) for ratio in font_ratios]

    # Find the mode (most common font ratio) - this is likely body text
    ratio_counts = Counter(rounded_ratios)
    mode_ratio = ratio_counts.most_common(1)[0][0]
    mode_count = ratio_counts.most_common(1)[0][1]

    # Calculate some statistics
    unique_ratios = len(set(rounded_ratios))

    # Check for predominantly bold content or numbered content
    total_bold = sum(1 for elem in elements if elem['is_bold'])
    total_starts_with_num = sum(
        1 for elem in elements if elem['starts_with_number'])
    bold_percentage = total_bold / original_count * 100
    numbered_percentage = total_starts_with_num / original_count * 100

    # Filter elements with new preservation logic
    font_filtered_elements = []
    preserved_by_bold = 0
    preserved_by_number = 0
    preserved_by_font_ratio = 0

    for elem in elements:
        should_preserve = False
        preserve_reason = ""

        # Standard font ratio filter - preserve elements with font ratio > mode_ratio
        if round(elem['font_size_ratio'], 3) > mode_ratio:
            should_preserve = True
            preserve_reason = "font_ratio"
            preserved_by_font_ratio += 1

        # New logic: preserve bold elements with font ratio equal to mode_ratio
        elif round(elem['font_size_ratio'], 3) == mode_ratio and elem['is_bold']:
            should_preserve = True
            # Check if it's also numbered for more specific counting
            if elem['starts_with_number']:
                preserve_reason = "numbered"
                preserved_by_number += 1
            else:
                preserve_reason = "bold"
                preserved_by_bold += 1

        if should_preserve:
            # Add preservation reason for debugging
            elem_copy = elem.copy()
            elem_copy['preservation_reason'] = preserve_reason
            font_filtered_elements.append(elem_copy)

    font_filtered_count = len(font_filtered_elements)

    if not font_filtered_elements:
        print("Warning: No elements remaining after font filtering!")
        return elements, {'stage': 'font_filtering_failed'}

    # Detect repeated elements (headers/footers) from font-filtered content
    repetition_info = detect_repeated_elements(
        font_filtered_elements, header_threshold=0.1, footer_threshold=0.9)
    repeated_indices = repetition_info['repeated_indices']
    repetition_stats = repetition_info['stats']
    # Show detected repeated patterns
    # Remove repeated elements
    final_elements = [
        elem for i, elem in enumerate(font_filtered_elements)
        if i not in repeated_indices
    ]

    final_count = len(final_elements)

    # Show final font ratios distribution
    if final_elements:
        final_ratios = [round(elem['font_size_ratio'], 3)
                        for elem in final_elements]
        final_ratio_counts = Counter(final_ratios)

        for ratio, count in sorted(final_ratio_counts.items(), reverse=True):
            percentage = count/final_count*100

        # Show preservation reasons distribution
        if 'preservation_reason' in final_elements[0]:
            preservation_reasons = [elem['preservation_reason']
                                    for elem in final_elements]
            reason_counts = Counter(preservation_reasons)

            for reason, count in reason_counts.items():
                percentage = count/final_count*100

    # Compile comprehensive statistics
    filter_stats = {
        'original_elements': original_count,
        'font_filtered_elements': font_filtered_count,
        'final_elements': final_count,
        'font_filtering_removed': original_count - font_filtered_count,
        'repetition_filtering_removed': font_filtered_count - final_count,
        'total_removed': original_count - final_count,
        'mode_ratio': mode_ratio,
        'mode_count': mode_count,
        'unique_ratios_original': unique_ratios,
        'unique_ratios_final': len(set(final_ratios)) if final_elements else 0,
        'repeated_patterns_detected': len(repetition_stats['repeated_patterns']),
        'total_pages': repetition_stats['total_pages'],
        'bold_percentage': bold_percentage,
        'numbered_percentage': numbered_percentage,
        'preserve_bold_numbered': False,  # No longer using this mechanism
        'preserved_by_bold': preserved_by_bold,
        'preserved_by_number': preserved_by_number,
        'preserved_by_font_ratio': preserved_by_font_ratio
    }

    return final_elements, filter_stats


def save_to_csv(elements, output_file, include_filter_stats=True):
    """Save extracted features to CSV file"""
    df = pd.DataFrame(elements)
    df.to_csv(output_file, index=False)
    print(f"\nFeatures saved to: {output_file}")


# Main execution
if __name__ == "__main__":
    # Replace with your PDF file path
    pdf_file = "pdfs/file05.pdf"

    try:
        # Extract features
        print("Starting PDF feature extraction...")
        features = extract_pdf_features(pdf_file)

        # Filter out body text and insignificant elements
        print("\nFiltering out body text and insignificant elements...")
        filtered_features, filter_stats = filter_significant_elements(features)

        if not filtered_features:
            print("\nWarning: No significant elements found after filtering!")
            print("This might happen if the PDF has very uniform formatting.")
            print("Saving original unfiltered data instead...")
            filtered_features = features

        # Save filtered results
        output_file = f"csv/pdf_features_filtered-{list(pdf_file.split('/'))[1]}.csv"
        save_to_csv(filtered_features, output_file)

    except Exception as e:
        print(f"Error processing PDF: {e}")
        print("Make sure:")
        print("1. PyMuPDF is installed: pip install PyMuPDF")
        print("2. PDF file path is correct")
        print("3. PDF file is not password protected")
