#!/usr/bin/env python3
"""
build_index.py

Pipeline:
1. Load plain text document (path provided)
2. Chunk text into 20+ chunks using sentence-accumulation + sliding overlap
3. Generate embeddings for each chunk using sentence-transformers all-mpnet-base-v2
4. Save chunks + embeddings to local chromadb vector store

Usage:
    python src\build_index.py --input data\hr_saas.txt --persist_dir .\chroma_store --collection_name hr_faq

Outputs:
 - Chromadb persisted DB in persist_dir
 - prints summary to stdout
"""

import argparse
import json
import os
import logging

from chroma_client import get_chroma_client, upsert_to_chroma
from utils import load_embedding_model, embed_texts, sentence_split, make_chunks

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def build_index(input_path: str, persist_dir: str, collection_name: str, model_name: str, approx_chars: int, overlap_chars: int):
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    sentences = sentence_split(text)
    logger.info("Split input into %d sentences/paragraphs", len(sentences))
    chunks = make_chunks(sentences, approx_chars=approx_chars, overlap_chars=overlap_chars)
    logger.info("Created %d chunks (approx_chars=%d overlap=%d)", len(chunks), approx_chars, overlap_chars)

    if len(chunks) < 20:
        logger.warning("Created fewer than 20 chunks (%d). Consider reducing approx_chars or overlap_chars.", len(chunks))

    # embed
    model = load_embedding_model(model_name)
    embeddings = embed_texts(model, chunks)

    # prepare ids & metadata
    ids = [f"chunk_{i:04d}" for i in range(len(chunks))]
    metadatas = [{"chunk_id": ids[i], "source": os.path.basename(input_path), "chunk_index": i} for i in range(len(chunks))]

    # chroma
    client = get_chroma_client(persist_dir)
    collection = upsert_to_chroma(client, collection_name, ids, chunks, metadatas, embeddings)

    # save manifest
    manifest = {
        "input_path": os.path.abspath(input_path),
        "collection_name": collection_name,
        "persist_dir": os.path.abspath(persist_dir),
        "n_chunks": len(chunks)
    }
    with open(os.path.join(persist_dir, f"{collection_name}_manifest.json"), "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)

    logger.info("Index build complete. Persist dir: %s", persist_dir)
    return manifest

def parse_args():
    parser = argparse.ArgumentParser(description="Build vector index from plain text for HR FAQ.")
    parser.add_argument("--input", "-i", required=True, help="Path to plain text input file")
    parser.add_argument("--persist_dir", "-p", default="./chroma_store", help="Directory where chroma DB will be persisted")
    parser.add_argument("--collection_name", "-c", default="hr_faq", help="Chromadb collection name")
    parser.add_argument("--model", default="all-mpnet-base-v2", help="SentenceTransformer model name")
    parser.add_argument("--approx_chars", type=int, default=800, help="Approx characters per chunk")
    parser.add_argument("--overlap_chars", type=int, default=200, help="Overlap characters between chunks")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.persist_dir, exist_ok=True)
    try:
        manifest = build_index(
            input_path=args.input,
            persist_dir=args.persist_dir,
            collection_name=args.collection_name,
            model_name=args.model,
            approx_chars=args.approx_chars,
            overlap_chars=args.overlap_chars
        )
        print(json.dumps(manifest, indent=2))
    except Exception as e:
        logger.exception("Failed to build index: %s", e)
        raise

if __name__ == "__main__":
    main()
