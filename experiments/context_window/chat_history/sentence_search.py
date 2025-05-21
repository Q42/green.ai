import json

import pandas as pd
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np

# Make sure NLTK's sentence tokenizer is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

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


def semantic_search_sentence_level(conversations, retention=0.2):
    # Extract user messages from each conversation
    documents = []
    for message in conversations[:-1]:
        if isinstance(message, dict) and 'content' in message:
            documents.append(message['content'])

    if not documents:
        return [], [], [], {'target_index': None, 'similarity_scores': {}, 'top_terms': {}}

    query = conversations[-1]['content']

    # Get the target document (last one)
    target_doc_idx = len(documents) - 1

    # Split each document into sentences
    all_sentences = []
    sentence_to_doc_map = {}
    doc_to_sentences_map = {}

    for doc_idx, doc in enumerate(documents):
        sentences = sent_tokenize(doc)
        doc_to_sentences_map[doc_idx] = []

        for sent_idx, sentence in enumerate(sentences):
            global_sent_idx = len(all_sentences)
            all_sentences.append(sentence)

            # Create sentence object with metadata
            sentence_obj = {
                'sentenceId': global_sent_idx,
                'documentId': doc_idx,
                'content': sentence
            }

            sentence_to_doc_map[global_sent_idx] = sentence_obj
            doc_to_sentences_map[doc_idx].append(global_sent_idx)

    # Split query into sentences
    query_sentences = sent_tokenize(query)

    # Embed all sentences (from documents and query)
    all_embeddings = embedder.encode(all_sentences + query_sentences, convert_to_tensor=True)

    # Separate document embeddings and query embeddings
    doc_sentence_embeddings = all_embeddings[:len(all_sentences)]
    query_sentence_embeddings = all_embeddings[len(all_sentences):]

    # Calculate similarity between each query sentence and all document sentences
    sentence_scores = {}
    for q_idx, q_embedding in enumerate(query_sentence_embeddings):
        # Calculate cosine similarity
        similarities = util.pytorch_cos_sim(q_embedding, doc_sentence_embeddings)[0]

        # Convert to numpy for easier manipulation
        similarities_np = similarities.cpu().numpy()

        # Update sentence scores (taking max score if a sentence is matched by multiple query sentences)
        for sent_idx, score in enumerate(similarities_np):
            if sent_idx not in sentence_scores or score > sentence_scores[sent_idx]:
                sentence_scores[sent_idx] = float(score)

    # Calculate document scores based on the max sentence score in each document
    doc_scores = {}
    for doc_idx in range(len(documents)):
        if doc_idx in doc_to_sentences_map:
            doc_sentence_indices = doc_to_sentences_map[doc_idx]
            if doc_sentence_indices:
                # Get the maximum sentence score for this document
                max_score = max([sentence_scores.get(sent_idx, 0) for sent_idx in doc_sentence_indices])
                doc_scores[doc_idx] = max_score
            else:
                doc_scores[doc_idx] = 0
        else:
            doc_scores[doc_idx] = 0

    # Sort documents by score
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)

    # Apply retention filter to documents
    allowed_count = max(1, int(len(documents) * retention))
    selected_doc_indices = [doc_idx for doc_idx, _ in sorted_docs[:allowed_count]]

    # Apply enhanced filter to document indices
    selected_doc_indices = enhanced_filter(selected_doc_indices, conversations, last_n=1, include_first=False)

    # Sort sentences by score
    sorted_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)

    # Filter sentences to only include those from selected documents
    selected_sentence_indices = [sent_idx for sent_idx, _ in sorted_sentences
                                 if sentence_to_doc_map[sent_idx]['documentId'] in selected_doc_indices]

    # Add all sentences from the first document (if any documents exist)
    if documents and 0 in doc_to_sentences_map:
        first_doc_sentences = doc_to_sentences_map[0]
        selected_sentence_indices.extend([sent_idx for sent_idx in first_doc_sentences
                                          if sent_idx not in selected_sentence_indices])

    # Add all sentences from the last document
    if documents and len(documents) - 1 in doc_to_sentences_map:
        last_doc_sentences = doc_to_sentences_map[len(documents) - 1]
        selected_sentence_indices.extend([sent_idx for sent_idx in last_doc_sentences
                                          if sent_idx not in selected_sentence_indices])

    # Create the list of sentence objects for the selected sentences
    selected_sentence_objects = [sentence_to_doc_map[sent_idx] for sent_idx in selected_sentence_indices]

    # Prepare additional information for analysis
    analysis_info = {
        'original_doc_length': len(documents),
        'new_doc_length': len(selected_doc_indices),
        'original_sentence_length': len(all_sentences),
        'new_sentence_length': len(selected_sentence_indices),
        'target_index': target_doc_idx,
        'doc_scores': {idx: score for idx, score in sorted_docs},
        'sentence_scores': {idx: score for idx, score in sorted_sentences[:20]}  # Top 20 for brevity
    }

    return list(selected_doc_indices), selected_sentence_indices, selected_sentence_objects, analysis_info


def reconstruct_conversation(original_conversation, selected_sentence_objects):
    # Group sentences by document ID
    sentences_by_doc = {}
    for sentence_obj in selected_sentence_objects:
        doc_id = sentence_obj['documentId']
        if doc_id not in sentences_by_doc:
            sentences_by_doc[doc_id] = []
        sentences_by_doc[doc_id].append(sentence_obj)

    # Sort sentences within each document by their sentence ID
    for doc_id in sentences_by_doc:
        sentences_by_doc[doc_id].sort(key=lambda x: x['sentenceId'])

    # Create a new conversation list
    new_conversation = []

    # Process each message in the original conversation (excluding the last query)
    for idx, message in enumerate(original_conversation[:-1]):
        if isinstance(message, dict) and 'content' in message and idx in sentences_by_doc:
            # Get all sentences for this document
            doc_sentences = sentences_by_doc[idx]

            # Combine the sentences into a single content string
            combined_content = " ".join([s['content'] for s in doc_sentences])

            # Create a new message with the same structure but updated content
            new_message = message.copy()
            new_message['content'] = combined_content

            # Only add the message if it has content
            if combined_content.strip():
                new_conversation.append(new_message)

    # Always include the last message (query) unchanged
    if original_conversation:
        new_conversation.append(original_conversation[-1])

    return new_conversation

if __name__ == '__main__':

    df = pd.read_csv("/Users/leenlaptop/Documents/repos/greenai/green.ai/datasets/benchmarks/MRCR/MRCR-64000.csv")

    prompt_str = df['prompt'].iloc[0]

    conversations = json.loads(prompt_str)

    # Find similar conversations to the last one
    doc_indices, sentence_indices, sentence_objects, analysis_info = semantic_search_sentence_level(
        conversations,
        retention=0.5,
    )


    print("=====begin======")
    # Use the indices to access the original conversations
    for idx in sorted(doc_indices):
        print(f"  {conversations[idx]['role']}: {conversations[idx]['content']}")
        print()
    print("=====end======")
    print(analysis_info)
    print(sorted(doc_indices))
