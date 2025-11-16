import os
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List
from tqdm import tqdm

# ----------- Embedding utility functions --------------

def load_embedding_model(model_name: str = "all-mpnet-base-v2"):
    # print("Loading embedding model: %s", model_name)
    model = SentenceTransformer(model_name)
    return model

def embed_query(model: SentenceTransformer, text: str):
    emb = model.encode([text], convert_to_numpy=True)[0]
    return emb.tolist()

def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int = 32) -> List[List[float]]:
    """Return list of vector embeddings as plain python lists."""
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i:i+batch_size]
        embs = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        for e in embs:
            embeddings.append(e.tolist())
    return embeddings

def save_embeddings_to_file(embeddings: List[List[float]], file_path: str):
    """Save embeddings to a .npz file."""
    np.savez_compressed(file_path, np.array(embeddings))
    # np.savez_compressed(file_path, np.array(embeddings))  # if embeddings not converted to np array during encoding
    print("Saved embeddings to %s", file_path)

# ------------ Chunking utility functions -----------

def sentence_split(text: str) -> List[str]:
    """sentence splitter using line breaks and punctuation."""
    # First split on double newlines (paragraphs), then split on periods/question/exclamation.
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    sentences = []
    for p in paragraphs:
        # keep punctuation by replacing with marker then splitting
        temp = p.replace("?", "?\n").replace("!", "!\n").replace(". ", ".\n")
        parts = [s.strip() for s in temp.splitlines() if s.strip()]
        sentences.extend(parts)
    return sentences

def make_chunks(sentences: List[str], approx_chars: int = 800, overlap_chars: int = 200) -> List[str]:
    """
    Accumulate sentences into chunks of approximately approx_chars characters.
    Overlap previous chunk by overlap_chars to provide context.
    """
    chunks = []
    current = []
    cur_len = 0
    i = 0
    while i < len(sentences):
        s = sentences[i]
        s_len = len(s)
        # If this single sentence is huge, still take it.
        if cur_len + s_len <= approx_chars or not current:
            current.append(s)
            cur_len += s_len
            i += 1
        else:
            chunk_text = " ".join(current).strip()
            if chunk_text:
                chunks.append(chunk_text)
            # create overlap: take from end of current until overlap_chars achieved
            overlap = []
            overlap_len = 0
            j = len(current) - 1
            while j >= 0 and overlap_len < overlap_chars:
                overlap.insert(0, current[j])
                overlap_len += len(current[j])
                j -= 1
            current = overlap.copy()
            cur_len = sum(len(x) for x in current)
    # final
    if current:
        last = " ".join(current).strip()
        if last:
            chunks.append(last)
    return chunks


def save_metrics_to_csv(metrics: dict, filename="outputs/metrics.csv"):
    import csv
    print("Saving metrics to CSV...")
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # Check if file exists to write header
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a', newline='') as csvfile:
        fieldnames = ["timestamp", "latency_ms", "tokens_prompt", "tokens_completion", "tokens_total", "estimated_cost_usd"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()  # Write header only once
        writer.writerow(metrics)
