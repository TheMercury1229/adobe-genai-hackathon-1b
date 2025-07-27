"""
Rule-based summarization for PDF sections using TF-IDF and sentence scoring.
Generates concise summaries without requiring LLMs for offline use.
"""

import re
import math
from collections import Counter
from typing import List, Dict, Any


def calculate_tf_idf_scores(texts: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Calculate TF-IDF scores for all words across all texts.

    Args:
        texts (List[str]): List of text documents

    Returns:
        Dict[str, Dict[str, float]]: TF-IDF scores for each word in each document
    """
    # Tokenize and clean texts
    documents = []
    for text in texts:
        # Simple tokenization and cleaning
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        # Filter out very short words and common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did',
                      'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        words = [word for word in words if len(
            word) > 2 and word not in stop_words]
        documents.append(words)

    # Calculate TF (Term Frequency)
    tf_scores = []
    for doc in documents:
        word_count = len(doc)
        tf = Counter(doc)
        tf_normalized = {word: count /
                         word_count for word, count in tf.items()}
        tf_scores.append(tf_normalized)

    # Calculate IDF (Inverse Document Frequency)
    all_words = set()
    for doc in documents:
        all_words.update(doc)

    idf_scores = {}
    total_docs = len(documents)
    for word in all_words:
        docs_containing_word = sum(1 for doc in documents if word in doc)
        idf_scores[word] = math.log(total_docs / docs_containing_word)

    # Calculate TF-IDF
    tfidf_scores = []
    for tf in tf_scores:
        tfidf = {}
        for word, tf_score in tf.items():
            tfidf[word] = tf_score * idf_scores[word]
        tfidf_scores.append(tfidf)

    return tfidf_scores


def score_sentence(sentence: str, tfidf_scores: Dict[str, float],
                   position_in_text: float, importance_keywords: List[str]) -> float:
    """
    Score a sentence based on multiple factors.

    Args:
        sentence (str): The sentence to score
        tfidf_scores (Dict[str, float]): TF-IDF scores for words
        position_in_text (float): Position of sentence in text (0-1)
        importance_keywords (List[str]): Keywords indicating importance

    Returns:
        float: Sentence score
    """
    words = re.findall(r'\b[a-zA-Z]+\b', sentence.lower())

    # TF-IDF score (average of word scores)
    tfidf_score = 0
    if words:
        word_scores = [tfidf_scores.get(word, 0) for word in words]
        tfidf_score = sum(word_scores) / len(word_scores)

    # Position score (beginning and end are more important)
    if position_in_text <= 0.3:  # First 30%
        position_score = 1.0
    elif position_in_text >= 0.7:  # Last 30%
        position_score = 0.8
    else:  # Middle
        position_score = 0.5

    # Length score (prefer medium-length sentences)
    word_count = len(words)
    if 10 <= word_count <= 25:
        length_score = 1.0
    elif 5 <= word_count < 10 or 25 < word_count <= 35:
        length_score = 0.8
    else:
        length_score = 0.3

    # Keyword importance score
    sentence_lower = sentence.lower()
    keyword_score = 0
    for keyword in importance_keywords:
        if keyword in sentence_lower:
            keyword_score += 1
    keyword_score = min(keyword_score * 0.3, 1.0)  # Cap at 1.0

    # Numeric/data presence (often important in documents)
    numeric_score = 0.2 if re.search(r'\d+', sentence) else 0

    # Combined score
    total_score = (
        tfidf_score * 0.4 +
        position_score * 0.25 +
        length_score * 0.2 +
        keyword_score * 0.1 +
        numeric_score * 0.05
    )

    return total_score


def extract_sentences(text: str) -> List[str]:
    """
    Extract sentences from text with basic sentence boundary detection.

    Args:
        text (str): Input text

    Returns:
        List[str]: List of sentences
    """
    # Simple sentence splitting with some improvements
    sentences = re.split(r'[.!?]+', text)

    # Clean and filter sentences
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        # Filter out very short sentences and those that are likely not complete
        if len(sentence) > 20 and len(sentence.split()) >= 4:
            cleaned_sentences.append(sentence)

    return cleaned_sentences


def generate_section_summary(section_text: str, tfidf_scores: Dict[str, float],
                             max_sentences: int = 3) -> str:
    """
    Generate a summary for a single section.

    Args:
        section_text (str): The section text
        tfidf_scores (Dict[str, float]): TF-IDF scores for the section
        max_sentences (int): Maximum number of sentences in summary

    Returns:
        str: Generated summary
    """
    sentences = extract_sentences(section_text)

    if len(sentences) <= max_sentences:
        return '. '.join(sentences) + '.'

    # Keywords that often indicate important information
    importance_keywords = [
        'important', 'key', 'main', 'primary', 'essential', 'crucial',
        'significant', 'notable', 'highlights', 'summary', 'conclusion',
        'benefits', 'advantages', 'features', 'best', 'top', 'recommended',
        'must', 'should', 'critical', 'major', 'fundamental'
    ]

    # Score sentences
    sentence_scores = []
    total_sentences = len(sentences)

    for i, sentence in enumerate(sentences):
        position = i / total_sentences if total_sentences > 1 else 0
        score = score_sentence(sentence, tfidf_scores,
                               position, importance_keywords)
        sentence_scores.append((sentence, score))

    # Sort by score and take top sentences
    sentence_scores.sort(key=lambda x: x[1], reverse=True)
    top_sentences = sentence_scores[:max_sentences]

    # Sort selected sentences by their original order in text
    original_order = []
    for selected_sentence, _ in top_sentences:
        original_index = sentences.index(selected_sentence)
        original_order.append((original_index, selected_sentence))

    original_order.sort(key=lambda x: x[0])
    summary_sentences = [sentence for _, sentence in original_order]

    return '. '.join(summary_sentences) + '.'


def create_refined_summaries(search_results: List[Dict[str, Any]],
                             max_sentences: int = 3) -> List[Dict[str, Any]]:
    """
    Create refined summaries for the top N most relevant sections.

    Args:
        search_results (List[Dict]): Results from semantic search function
        max_sentences (int): Maximum sentences per summary (2-3 recommended)

    Returns:
        List[Dict]: Refined results with summaries added
    """
    if not search_results:
        return []

    # Extract all section texts for TF-IDF calculation
    all_texts = [result['section_text'] for result in search_results]

    # Calculate TF-IDF scores across all sections
    tfidf_scores_list = calculate_tf_idf_scores(all_texts)

    # Generate summaries for each section
    refined_results = []

    for i, result in enumerate(search_results):
        section_text = result['section_text']
        tfidf_scores = tfidf_scores_list[i] if i < len(
            tfidf_scores_list) else {}

        # Generate summary
        summary = generate_section_summary(
            section_text, tfidf_scores, max_sentences)

        # Create refined result
        refined_result = {
            'summary': summary,
            'section_title': result.get('section_title', 'Untitled'),
            'filename': result.get('filename', 'Unknown'),
            'page': result.get('start_page', 0),
            'similarity_score': result.get('similarity_score', 0.0)
        }

        refined_results.append(refined_result)

    return refined_results


def print_refined_summaries(refined_results: List[Dict[str, Any]]):
    """
    Pretty print the refined summaries.

    Args:
        refined_results (List[Dict]): Results with summaries
    """
    if not refined_results:
        print("No summaries available.")
        return

    print("\n" + "="*80)
    print("REFINED SECTION SUMMARIES")
    print("="*80)

    for i, result in enumerate(refined_results, 1):
        print(f"\n{i}. {result['section_title']}")
        print(f"   File: {result['filename']} (Page {result['page']})")
        print(f"   Similarity Score: {result['similarity_score']:.4f}")
        print(f"   Summary:")
        # Format summary with proper indentation
        summary_lines = result['summary'].split('. ')
        for line in summary_lines:
            if line.strip():
                print(f"   • {line.strip()}.")
        print("-" * 60)


# Integration example
def example_usage():
    """
    Example of how to use the summarization function with search results.
    """
    # Example search results (would come from your semantic search function)
    example_results = [
        {
            'filename': 'South of France - Cuisine.pdf',
            'section_title': 'Famous Dishes',
            'start_page': 3,
            'similarity_score': 0.8542,
            'section_text': 'The South of France is renowned for its Mediterranean cuisine featuring fresh seafood, olive oil, and aromatic herbs. Bouillabaisse is the most famous dish from Marseille, a traditional fish stew that combines various Mediterranean fish with saffron and herbs. The dish must be prepared with specific types of fish including scorpion fish, sea robin, and conger eel. Another essential dish is Ratatouille, originating from Nice, which combines eggplant, zucchini, bell peppers, tomatoes, and herbs. The key to authentic Ratatouille is cooking each vegetable separately before combining them. Socca, a chickpea pancake from Nice, is a popular street food that represents the Italian influence in the region. These dishes showcase the diversity and richness of Southern French culinary traditions.'
        },
        {
            'filename': 'South of France - Cuisine.pdf',
            'section_title': 'Must-Visit Restaurants',
            'start_page': 4,
            'similarity_score': 0.7821,
            'section_text': 'For an authentic culinary experience, visit these renowned establishments in the South of France. La Petite Maison in Nice offers exceptional Mediterranean dishes with a focus on fresh, local ingredients and traditional recipes. The restaurant is famous for its bouillabaisse and fresh seafood selections. In Marseille, Chez Fonfon provides an authentic local dining experience with traditional Provençal cuisine. Their specialty is the classic bouillabaisse served with rouille sauce and crusty bread. L\'Oustau de Baumanière in Les Baux-de-Provence is a Michelin-starred restaurant that elevates traditional Southern French cuisine to gourmet levels. The restaurant sources ingredients from local farms and creates innovative interpretations of classic dishes. These establishments represent the pinnacle of Southern French dining and offer visitors an unforgettable gastronomic journey.'
        }
    ]

    # Generate refined summaries
    refined_summaries = create_refined_summaries(
        example_results, max_sentences=3)

    # Print results
    print_refined_summaries(refined_summaries)

    return refined_summaries


if __name__ == "__main__":
    # Run the example
    example_usage()
