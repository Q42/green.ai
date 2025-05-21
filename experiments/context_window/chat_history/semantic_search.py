import json
import pandas as pd
from sentence_transformers import SentenceTransformer, util

embedder = SentenceTransformer("all-MiniLM-L6-v2")

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

def semantic_search(conversations, retention=0.2):
    # Extract user messages from each conversation
    documents = []
    for message in conversations[:-1]:
        if isinstance(message, dict) and 'content' in message:
            documents.append(message['content'])

    if not documents:
        return [], {'target_index': None, 'similarity_scores': {}, 'top_terms': {}}

    query = conversations[-1]['content']

    # Get the target document (last one)
    target_doc_idx = len(documents) - 1

    all_indices = list(range(len(documents)))

    allowed_count = int(len(documents) * retention)

    corpus_embeddings = embedder.encode(documents, convert_to_tensor=True)
    #corpus_embeddings = corpus_embeddings.to("cuda")
    #corpus_embeddings = util.normalize_embeddings(corpus_embeddings)

    query_embeddings = embedder.encode(query, convert_to_tensor=True)
    #query_embeddings = query_embeddings.to("cuda")
    #query_embeddings = util.normalize_embeddings(query_embeddings)

    hits = util.semantic_search(query_embeddings, corpus_embeddings, score_function=util.dot_score, top_k=allowed_count)

    if allowed_count == 0:
        selected_indices = []
    else:
        selected_indices = sorted([d['corpus_id'] for d in hits[0]])

    # Sort by similarity (highest first)
    selected_indices = enhanced_filter(selected_indices, conversations, last_n=4, include_first=False)

    # Prepare additional information for analysis
    analysis_info = {
        'original length': len(documents),
        'new_length': len(selected_indices),
        'target_index': target_doc_idx,
    }

    return selected_indices, analysis_info

def semantic_search_variable(conversations, sensitivity=0.2):
    # Extract user messages from each conversation
    documents = []
    for message in conversations[:-1]:
        if isinstance(message, dict) and 'content' in message:
            documents.append(message['content'])

    if not documents:
        return [], {'target_index': None, 'similarity_scores': {}, 'top_terms': {}}

    query = conversations[-1]['content']

    # Get the target document (last one)
    target_doc_idx = len(documents) - 1

    all_indices = list(range(len(documents)))

    corpus_embeddings = embedder.encode(documents, convert_to_tensor=True)
    #corpus_embeddings = corpus_embeddings.to("cuda")
    #corpus_embeddings = util.normalize_embeddings(corpus_embeddings)

    query_embeddings = embedder.encode(query, convert_to_tensor=True)
    #query_embeddings = query_embeddings.to("cuda")
    #query_embeddings = util.normalize_embeddings(query_embeddings)

    hits = util.semantic_search(query_embeddings, corpus_embeddings, score_function=util.dot_score)

    filtered_hits = [hit for hit in hits[0] if hit['score'] >= sensitivity]

    selected_indices = sorted([d['corpus_id'] for d in filtered_hits])

    # Sort by similarity (highest first)
    selected_indices = enhanced_filter(selected_indices, conversations, last_n=4, include_first=False)

    # Prepare additional information for analysis
    analysis_info = {
        'original length': len(documents),
        'new_length': len(selected_indices),
        'target_index': target_doc_idx,
    }

    return selected_indices, analysis_info


if __name__ == '__main__':

    df = pd.read_csv("/Users/leenlaptop/Documents/repos/greenai/green.ai/datasets/benchmarks/MRCR/MRCR-64000.csv")

    prompt_str = df['prompt'].iloc[0]

    conversations = json.loads(prompt_str)

    # Find similar conversations to the last one
    similar_indices, analysis_info = semantic_search_variable(
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
    print(sorted(similar_indices))
