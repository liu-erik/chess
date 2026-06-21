"""
basic_rag.py

System A: the "baseline" approach.

This is the simplest possible RAG (Retrieval Augmented Generation) pipeline,
now operating on REAL detected blunders from a chess game (found by Stockfish
in game_analyzer.py) instead of free-text questions:

1. Take a detected blunder (move, eval swing, position)
2. Turn it into a natural language description
3. Embed that description and find the most similar coaching chunks
4. Stuff those chunks into a prompt and ask Claude to explain the mistake

No agents, no multi-step reasoning - just retrieve and answer in one shot.
This is our "control group" that we'll compare the multi-agent system against.
"""

import os
import time
from anthropic import Anthropic
from knowledge_base import get_openai_client, embed_text, get_collection
from dotenv import load_dotenv

load_dotenv()


def retrieve_relevant_chunks(question, n_results=3):
    """
    Given a question, find the n_results most semantically similar chess chunks.

    How this works: we embed the question into the same vector space as our
    knowledge base, then ChromaDB does a similarity search (cosine similarity
    under the hood) to find the closest vectors.
    """
    openai_client = get_openai_client()
    collection = get_collection()

    question_vector = embed_text(openai_client, question)

    results = collection.query(
        query_embeddings=[question_vector],
        n_results=n_results
    )

    # results["documents"][0] is a list of the actual chunk text strings
    return results["documents"][0]


def blunder_to_question(blunder):
    """
    Converts a detected blunder (from game_analyzer.py) into a natural
    language description we can use for retrieval and prompting. This is
    the bridge between "Stockfish found a mistake" and "explain this mistake
    to a student."
    """
    return (
        f"In a game, {blunder['player']} played {blunder['move']} on move "
        f"{blunder['move_number']}, which lost about "
        f"{blunder['swing_centipawns'] / 100:.1f} pawns of advantage "
        f"(evaluation went from {blunder['eval_before']} to {blunder['eval_after']}). "
        f"What kind of mistake is this and what should the player have considered instead?"
    )


def answer_with_basic_rag(blunder):
    """
    The full System A pipeline: convert the blunder to a description,
    retrieve relevant chunks, then ask Claude to explain the mistake.

    Returns a dict with the answer and how long it took (we need latency
    for our benchmark comparison later).
    """
    start_time = time.time()

    question = blunder_to_question(blunder)
    chunks = retrieve_relevant_chunks(question)
    context = "\n\n".join(chunks)

    client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    prompt = f"""You are a chess coach. Use the following chess knowledge to explain a mistake
a student made in their game.

CHESS KNOWLEDGE:
{context}

THE MISTAKE:
{question}

Give a clear, helpful explanation of what went wrong based on the knowledge provided above."""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )

    elapsed = time.time() - start_time

    return {
        "answer": response.content[0].text,
        "latency_seconds": round(elapsed, 2),
        "retrieved_chunks": chunks
    }


if __name__ == "__main__":
    # Quick manual test with a fake blunder (normally this comes from game_analyzer.py)
    test_blunder = {
        "move_number": 7,
        "player": "Black",
        "move": "Qh4",
        "eval_before": 20,
        "eval_after": -180,
        "swing_centipawns": 200,
    }
    result = answer_with_basic_rag(test_blunder)
    print("BLUNDER:", blunder_to_question(test_blunder))
    print("\nANSWER:", result["answer"])
    print(f"\nLatency: {result['latency_seconds']}s")
