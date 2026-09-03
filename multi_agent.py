"""
multi_agent.py

System B: the LangGraph multi-agent approach, operating on REAL detected
blunders from a chess game (found by Stockfish in game_analyzer.py).

Instead of one retrieval + one answer (like basic_rag.py), this breaks the task
into 3 specialized steps, each handled by a focused "agent" (really just a
focused prompt + LLM call), coordinated by LangGraph as a state machine:

  1. Retrieval Agent  - same semantic search as System A, based on the blunder
  2. Analysis Agent    - classifies WHAT TYPE of mistake this was (tactical
                         oversight, positional error, opening principle
                         violation, etc.) and what the player likely missed
  3. Coach Agent       - takes the analysis + retrieved knowledge and writes a
                         personalized, specific coaching response tied to the
                         exact position and move played

This is "multi-agent" because each node has a distinct role and they pass
state to each other, rather than one model doing everything in a single pass.
"""

import os
import time
from typing import TypedDict, List
from anthropic import Anthropic
from langgraph.graph import StateGraph, END
from basic_rag import retrieve_relevant_chunks, blunder_to_question
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


class ChessState(TypedDict):
    blunder_description: str
    blunder_data: dict
    retrieved_chunks: List[str]
    analysis: str
    final_answer: str


def retrieval_agent(state: ChessState) -> ChessState:
    """Fetches relevant chess knowledge based on the blunder description."""
    chunks = retrieve_relevant_chunks(state["blunder_description"])
    state["retrieved_chunks"] = chunks
    return state


def analysis_agent(state: ChessState) -> ChessState:
    """
    Instead of jumping straight to an answer, this agent first reasons about
    WHAT TYPE of mistake this specific move was - tactical (missed a fork/pin),
    positional (weakened pawn structure), or principle-based (moved queen too
    early, ignored development). This extra reasoning step is what
    differentiates the multi-agent approach from basic RAG.
    """
    context = "\n\n".join(state["retrieved_chunks"])
    blunder = state["blunder_data"]

    prompt = f"""You are a chess analysis expert. A player made a mistake that lost
{blunder['swing_centipawns'] / 100:.1f} pawns of evaluation. Given the move played and
relevant chess knowledge below, identify:
1. What TYPE of mistake this was (tactical oversight, positional error, opening
   principle violation, or calculation error)
2. What the player likely missed or miscalculated when making this move
3. What level of player (beginner/intermediate/advanced) commonly makes this mistake

CHESS KNOWLEDGE:
{context}

THE MISTAKE:
{state['blunder_description']}

Provide a brief analysis (3-4 sentences) covering the points above."""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )

    state["analysis"] = response.content[0].text
    return state


def coach_agent(state: ChessState) -> ChessState:
    """
    Takes the retrieved knowledge AND the analysis (what TYPE of mistake this
    was) and synthesizes a final, personalized coaching response tied to the
    exact move and position. This is richer than System A's answer because
    it's informed by the analysis step's classification of the mistake type.
    """
    context = "\n\n".join(state["retrieved_chunks"])

    prompt = f"""You are an expert chess coach giving personalized advice on a specific
mistake from a player's actual game.

CHESS KNOWLEDGE:
{context}

ANALYSIS OF WHAT TYPE OF MISTAKE THIS WAS:
{state['analysis']}

THE MISTAKE:
{state['blunder_description']}

Using the analysis above, give a specific, actionable coaching response. Don't just
restate facts - explain WHY this particular move was a mistake and give a concrete
tip for recognizing and avoiding this type of error in future games."""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}]
    )

    state["final_answer"] = response.content[0].text
    return state


def build_graph():
    """
    Wires the three agents together into a linear pipeline:
    retrieval -> analysis -> coach -> end
    """
    graph = StateGraph(ChessState)

    graph.add_node("retrieval", retrieval_agent)
    graph.add_node("analysis", analysis_agent)
    graph.add_node("coach", coach_agent)

    graph.set_entry_point("retrieval")
    graph.add_edge("retrieval", "analysis")
    graph.add_edge("analysis", "coach")
    graph.add_edge("coach", END)

    return graph.compile()


def answer_with_multi_agent(blunder):
    """
    Runs the full multi-agent pipeline on a detected blunder and returns the
    answer + latency, in the same format as basic_rag.answer_with_basic_rag
    so we can compare them directly in eval.py
    """
    start_time = time.time()

    description = blunder_to_question(blunder)

    app = build_graph()
    initial_state: ChessState = {
        "blunder_description": description,
        "blunder_data": blunder,
        "retrieved_chunks": [],
        "analysis": "",
        "final_answer": ""
    }

    final_state = app.invoke(initial_state)

    elapsed = time.time() - start_time

    return {
        "answer": final_state["final_answer"],
        "latency_seconds": round(elapsed, 2),
        "retrieved_chunks": final_state["retrieved_chunks"],
        "analysis": final_state["analysis"]
    }
  

if __name__ == "__main__":
    # Quick manual test - compare this output to basic_rag.py's output
    # for the same blunder to see the difference in depth
    test_blunder = {
        "move_number": 7,
        "player": "Black",
        "move": "Qh4",
        "eval_before": 20,
        "eval_after": -180,
        "swing_centipawns": 200,
    }
    result = answer_with_multi_agent(test_blunder)
    print("BLUNDER:", blunder_to_question(test_blunder))
    print("\nANALYSIS:", result["analysis"])
    print("\nFINAL ANSWER:", result["answer"])
    print(f"\nLatency: {result['latency_seconds']}s")
