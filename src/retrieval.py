# src/retrieval.py
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import openai

# -------------------
# Load Chunks
# -------------------
with open('outputs/d2l_chunks.json', 'r', encoding='utf-8') as f:
    chunks = json.load(f)

# -------------------
# Create Embeddings
# -------------------
model = SentenceTransformer('all-MiniLM-L6-v2')
chunk_texts = [chunk['text'] for chunk in chunks]
chunk_embeddings = model.encode(chunk_texts, show_progress_bar=True)

# -------------------
# Retrieval Function
# -------------------
def retrieve_relevant_chunks(query, k=3):
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, chunk_embeddings)[0]
    top_k_idx = np.argsort(similarities)[-k:][::-1]
    return [chunks[i] for i in top_k_idx]

# -------------------
# Explanation Function
# -------------------
openai.api_key = "YOUR_OPENAI_KEY"

def generate_explanation(question, chunks):
    context = "\n\n".join([chunk['text'] for chunk in chunks])
    prompt = f"""
You are a Deep Learning tutor explaining concepts to a beginner.

The student asked: "{question}"

Textbook excerpts:
{context}

Explain this in **clear, simple language**.
Use analogies or examples if helpful.
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5
    )
    return response.choices[0].message['content']

# -------------------
# Test
# -------------------
if __name__ == "__main__":
    question = "Explain backpropagation"
    top_chunks = retrieve_relevant_chunks(question)
    answer = generate_explanation(question, top_chunks)
    print("\n🧠 Explanation:\n", answer)
