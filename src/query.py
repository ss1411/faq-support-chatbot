#!/usr/bin/env python3
"""
query.py

Pipeline:
1. Accept a user question (CLI argument or interactive input)
2. Convert question to embeddings using same local model
3. Perform vector search (k-NN) on chromadb
4. Retrieve relevant chunks (documents + metadata)
5. Call OpenAI (gpt-4o-mini) to generate an answer using the retrieved chunks as context
6. Print/return JSON with user_question, system_answer, chunks_related

Usage:
    python src\query.py --question "key steps involved in employee onboarding process?" --persist_dir ./chroma_store --collection_name hr_faq --k 2
"""

import argparse
import os
import json
import time
import logging
from typing import List, Dict
from pydantic import BaseModel

from dotenv import load_dotenv
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)
openrouter_api_key = os.getenv('OPENROUTER_API_KEY')

from evaluator_agent import evaluate_response
from chroma_client import get_chroma_client, search_chroma
from llm_client import OpenAIClient
from utils import load_embedding_model, embed_query, save_metrics_to_csv

openai_client = OpenAIClient(openrouter_api_key)

# set up file logging along with console logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler(__name__ + '.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(logging.StreamHandler())


def prompt_for_answer(question: str, retrieved_chunks: List[Dict], system_instructions: str = None) -> str:
    """
    Construct a prompt that gives the model the retrieved chunks and asks for a concise, accurate answer.
    """
    system_instructions = system_instructions or ("You are an HR SaaS product assistant. Use only the provided chunks to answer the user. "
                                                "When appropriate, cite the chunk_id(s) you used in brackets like [chunk_0001]. If the answer is not found, say you don't know and suggest next steps.")
    # build context
    context_pieces = []
    for i, c in enumerate(retrieved_chunks):
        meta = c.get("metadata", {})
        chunk_id = meta.get("chunk_id", f"chunk_{i}")
        context_pieces.append(f"--- {chunk_id} ---\n{c.get('document')}\n")
    context_text = "\n".join(context_pieces)

    # call OpenAI endpoint
    try:
        response, input_tokens, output_tokens = openai_client.call_openai(user_question=question, context=context_text, system_msg=system_instructions)
    except Exception as e:
        logger.exception("OpenAI call failed: %s", e)
        raise
    cost_for_request = openai_client.calculate_request_cost(input_tokens, output_tokens)
    logger.info(f"Input Tokens: {input_tokens}\nOutput Tokens: {output_tokens}")
    logger.info(f"Request Cost(USD): ${cost_for_request:.6f}")
    return response, input_tokens, output_tokens, cost_for_request


#-------- CLI / main --------------

def parse_args():
    parser = argparse.ArgumentParser(description="Query the HR FAQ vector store and get an LLM answer.")
    parser.add_argument("--question", "-q", type=str, help="User question. If not provided, will prompt interactively.")
    parser.add_argument("--persist_dir", "-p", default="./chroma_store", help="Chromadb persist directory")
    parser.add_argument("--collection_name", "-c", default="hr_faq", help="Chromadb collection name")
    parser.add_argument("--model", default="all-mpnet-base-v2", help="Embedding model (sentence-transformers)")
    parser.add_argument("--k", type=int, default=5, help="Number of nearest neighbors to retrieve")
    return parser.parse_args()

def main():
    args = parse_args()
    question = args.question
    if not question:
        question = input("Enter your question: ").strip()
    if not question:
        print("No question provided. Exiting.")
        return

    if not os.path.isdir(args.persist_dir):
        raise FileNotFoundError(f"Persist directory not found: {args.persist_dir}. Have you run build_index.py?")

    start = time.perf_counter()
    # load model
    model = load_embedding_model(args.model)
    emb = embed_query(model, question)

    # connect chroma & search
    client = get_chroma_client(args.persist_dir)
    try:
        retrieved = search_chroma(client, args.collection_name, emb, k=args.k)
    except Exception as e:
        logger.exception("Chromadb query failed: %s", e)
        raise

    # generate answer with LLM
    try:
        answer, num_input_tokens, num_output_tokens, cost = prompt_for_answer(question, retrieved)
    except Exception as e:
        logger.exception("Answer generation failed: %s", e)
        raise

    latency_ms = (time.perf_counter() - start) * 1000
    metrics = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
            "latency_ms": "{:.6f}".format(latency_ms),
            "tokens_prompt": num_input_tokens,
            "tokens_completion": num_output_tokens,
            "tokens_total": num_input_tokens + num_output_tokens,
            "estimated_cost_usd": "{:.6f}".format(cost)
        }
        
    logger.info(f"Request metrics: {metrics}")
    save_metrics_to_csv(metrics)

    output = {
        "user_question": question,
        "system_answer": answer,
        "chunks_related": retrieved
    }

    print("----- Final Output -----\n")
    print(json.dumps(output, indent=2, ensure_ascii=False))

    # response evaluation
    evaluation = evaluate_response(question, answer, retrieved)
    print("\n----- Evaluation -----\n")
    print(json.dumps(evaluation, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
