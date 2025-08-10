import re
from typing import List, Dict

# Helper function to count tokens (approximate: split by whitespace) -- Replace with tokenizer if needed
def count_tokens(text: str) -> int:
    return len(text.split())

def chunk_document(text: str, max_tokens: int = 512, overlap: int = 200) -> List[str]:
    words = text.split()
    i = 0
    chunks = []
    total_words = len(words)
    while i < total_words:
        # Get the end index for the chunk
        end = min(i + max_tokens, total_words)
        chunk = words[i:end]
        chunks.append(' '.join(chunk))
        # Move forward (overlap)
        if end == total_words:
            break
        i += max_tokens - overlap
    return chunks

def process_documents(docs: List[Dict], max_tokens: int = 512, overlap: int = 200) -> List[Dict]:
    chunked_docs = []
    for doc in docs:
        chunks = chunk_document(doc['content'], max_tokens, overlap)
        for idx, chunk in enumerate(chunks):
            # Attach metadata
            chunk_metadata = {
                'chunk_id': f"{doc.get('id','unk')}-{idx}",
                'category': doc.get('category', None),
                'priority': doc.get('priority', None),
                'date': doc.get('date', None),
            }
            chunked_docs.append({'text': chunk, 'metadata': chunk_metadata})
    return chunked_docs
