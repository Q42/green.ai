from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import pandas as pd


def find_similar_documents(conversations, sensitivity=0.2, top_n=100):
    # Extract user messages from each conversation
    documents = []
    for message in conversations:
        if isinstance(message, dict) and 'content' in message:
            documents.append(message['content'])

    # Create TF-IDF vectors
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()

    # Get the target document (last one)
    target_doc_idx = len(documents) - 1

    # Calculate cosine similarity between target and all documents
    similarities = cosine_similarity(tfidf_matrix, tfidf_matrix[target_doc_idx])

    # Flatten the similarities array
    similarities = similarities.flatten()

    # Get all document indices except the target itself
    all_indices = list(range(len(documents)))
    all_indices.remove(target_doc_idx)

    # If there are no other documents, return empty results
    if not all_indices:
        return [], {'target_index': target_doc_idx, 'similarity_scores': {}, 'top_terms': {}}

    # Find the min and max similarity scores (excluding the target document)
    min_sim = min(similarities[all_indices])
    max_sim = max(similarities[all_indices])

    if min_sim == max_sim:
        # If all documents have the same similarity, use a binary approach
        # sensitivity=0 returns all, sensitivity>0 returns none
        threshold = min_sim if sensitivity == 0 else min_sim + 0.0001
    else:
        # Map sensitivity from [0,1] to [min_sim, max_sim]
        # When sensitivity=0, threshold=min_sim (returns all docs)
        # When sensitivity=1, threshold=max_sim (returns only exact matches)
        threshold = min_sim + sensitivity * (max_sim - min_sim)

    # Find documents above the threshold (excluding the target document itself)
    similar_indices = [idx for idx in all_indices if similarities[idx] >= threshold]

    # Sort by similarity (highest first)
    similar_indices = sorted(similar_indices, key=lambda idx: similarities[idx], reverse=True)

    # Limit to top_n results
    similar_indices = similar_indices[:top_n]
    # Prepare additional information for analysis
    analysis_info = {
        'target_index': target_doc_idx,
        'similarity_scores': {idx: similarities[idx] for idx in similar_indices},
        'top_terms': {}
    }

    # For each similar document, find the top common terms
    for idx in similar_indices:
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

    return similar_indices, analysis_info
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
        sensitivity=0.2,
        top_n=100
    )

    # Use the indices to access the original conversations
    for idx in similar_indices:
        print(f"Similar conversation found at index {idx}:")
        print(f"  {conversations[idx]['role']}: {conversations[idx]['content']}")
        print()