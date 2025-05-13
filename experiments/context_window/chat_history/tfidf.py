from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import pandas as pd


def enhanced_filter(indexes, words, last_n=2, include_first=True):
    # Create enhanced indexes list
    enhanced_indexes = set(indexes)  # Original indexes

    # Add first index if requested
    if include_first:
        enhanced_indexes.add(0)

    # Add last n indexes
    for i in range(max(0, len(words) - last_n), len(words)):
        enhanced_indexes.add(i)

    # Filter valid indexes, sort them, and get the corresponding words
    valid_indexes = sorted([i for i in enhanced_indexes if 0 <= i < len(words)])

    return set(sorted(valid_indexes))

def find_similar_documents(conversations, sensitivity=0.2):
    # Extract user messages from each conversation
    documents = []
    for message in conversations:
        if isinstance(message, dict) and 'content' in message:
            documents.append(message['content'])

    if not documents:
        return [], {'target_index': None, 'similarity_scores': {}, 'top_terms': {}}

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    # Get the target document (last one)
    target_doc_idx = len(documents) - 1

    # Calculate cosine similarity between target and all documents
    similarities = cosine_similarity(tfidf_matrix, tfidf_matrix[target_doc_idx])
    similarities = similarities.flatten()

    # Get all document indices except the target itself
    all_indices = list(range(len(documents)))
    all_indices.remove(target_doc_idx)

    # Sort indices by similarity descending (most relevant first)
    sorted_indices = sorted(all_indices, key=lambda idx: similarities[idx], reverse=True)

    # Determine the allowed count based on sensitivity.
    # A sensitivity of 1 returns all documents,
    # sensitivity of 0 returns 0 documents from similarity (only enhanced_filter would later be applied)
    allowed_count = int(len(sorted_indices) * sensitivity)
    # In case sensitivity is so low that allowed_count becomes 0, you might directly use the enhanced filter result.
    if allowed_count == 0:
        selected_indices = []
    else:
        selected_indices = sorted_indices[:allowed_count]

    # Sort by similarity (highest first)
    selected_indices = enhanced_filter(selected_indices, documents, last_n=4, include_first=False)

    # Prepare additional information for analysis
    analysis_info = {
        'original length': len(documents),
        'new_length': len(selected_indices),
        'target_index': target_doc_idx,
        'for ': {idx: similarities[idx] for idx in selected_indices},
        'top_terms': {}
    }

    # For each similar document, find the top common terms
    for idx in selected_indices:
        # Get the document vectors
        target_vector = tfidf_matrix[target_doc_idx].toarray().flatten()
        doc_vector = tfidf_matrix[idx].toarray().flatten()

        # Find the top common terms
        common_importance = target_vector * doc_vector
        top_term_indices = common_importance.argsort()[-5:][::-1]  # Top 5 terms

        # Get the actual terms
        top_terms = [(feature_names[i], float(common_importance[i])) for i in top_term_indices if
                     common_importance[i] > 0]
        analysis_info['top_terms'][idx] = top_terms

    return selected_indices, analysis_info
# Example usage:
# Assuming tfidf_matrix and feature_names are already defined
# and the last document index is len(tfidf_matrix) - 1

if __name__ == '__main__':

    df = pd.read_csv("/Users/leenlaptop/Documents/repos/greenai/green.ai/datasets/benchmarks/MRCR/MRCR-64000.csv")

    prompt_str = df['prompt'].iloc[0]

    conversations = json.loads(prompt_str)

    # Find similar conversations to the last one
    similar_indices, analysis_info = find_similar_documents(
        conversations,
        sensitivity=0.5,
    )

    print("=====begin======")
    # Use the indices to access the original conversations
    for idx in sorted(similar_indices):
        print(f"  {conversations[idx]['role']}: {conversations[idx]['content']}")
        print()
    print("=====end======")
    print(analysis_info)