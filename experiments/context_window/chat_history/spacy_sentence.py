import spacy
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity  # Efficient cosine similarity for vectors

# Load spaCy model (prefer a model with word vectors if possible, like en_core_web_md)
def initialize_nlp():
    return spacy.load("en_core_web_sm")

nlp = initialize_nlp()

def normalize(text):
    return ' '.join(text.lower().split())


def tokenize_sentences(text):
    # Get sentences using spaCy's sentencizer
    doc = nlp(text)
    return [sent.text for sent in doc.sents]


def compute_vector(sent):
    # Compute vector using spaCy
    return nlp(sent).vector


def is_redundant_vector(new_vector, previous_vectors, sim_threshold=0.9):
    # If no previous sentence, it's not redundant
    if len(previous_vectors) == 0:
        return False
    # Compute cosine similarities between new_vector and all previous_vector(s)
    # Reshape to 2D for sklearn
    similarities = cosine_similarity(new_vector.reshape(1, -1), np.array(previous_vectors))
    return np.any(similarities > sim_threshold)


def summarize_sentences(sentences, previous_vectors, sim_threshold=0.9):
    summarized_sentences = []
    redundant_streak = False

    # Process vectors in batch to avoid repeated processing:
    # This may help if you have a very large batch, here we process one by one, but the vector computation inside
    # can be batched when integrated with an ANN or similar.
    for sent in sentences:
        # Compute vector
        new_vector = compute_vector(sent)  # Could be batched if needed.

        if is_redundant_vector(new_vector, previous_vectors, sim_threshold):
            if not redundant_streak:
                summarized_sentences.append("[referenced earlier]")
                redundant_streak = True
        else:
            previous_vectors.append(new_vector)
            summarized_sentences.append(sent)
            redundant_streak = False

    return summarized_sentences


def summarize_chat(chat, sim_threshold=0.9):
    previous_vectors = []
    summarized_chat = []
    all_sentences = []
    # First, normalize and separate sentences for each message
    messages_sentences = []
    for message in chat:
        norm_text = normalize(message["content"])
        sentences = tokenize_sentences(norm_text)
        messages_sentences.append(sentences)
        all_sentences.extend(sentences)

    # Use nlp.pipe to process all sentences at once for vectorization if needed:
    # However, since we have a dependency between messages (previous vectors), we'll loop through messages.
    for i, sentences in enumerate(messages_sentences):
        summarized = summarize_sentences(sentences, previous_vectors, sim_threshold)
        # Reassemble sentences back into the content string.
        summarized_content = " ".join(summarized)
        summarized_chat.append({
            "role": chat[i]["role"],
            "content": summarized_content
        })

    return summarized_chat

if __name__ == "__main__":
    chat_input = [
        {"role": "user", "content": "Hello, I need help with my account. I need help with my account."},
        {"role": "system", "content": "Sure, I can help. Please tell me more about your account."},
        {"role": "user", "content": "I need help with my account and I forgot my password."},
        {"role": "system", "content": "Sure, I can help. Please tell me more about your situation. Wubadubadubdub"},
    ]

    print("\n\nInput\n")
    print(chat_input)

    summarized_chat = summarize_chat(chat_input)
    print("\n\nOutput\n")
    print(summarized_chat)