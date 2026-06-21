"""
eval.py

This is the heart of the project: a benchmark that runs detected blunders
from real chess games through both System A (basic RAG) and System B
(multi-agent), then uses Claude as an "LLM judge" to score each answer on
relevance and specificity.

Blunders are found using Stockfish (game_analyzer.py) - this is what makes
the evaluation grounded in real chess mistakes rather than made-up questions.

Why LLM-as-judge? It's an established evaluation technique when you don't have
human graders available - you ask a strong LLM to act as an impartial judge and
score outputs against criteria.

Output: a results.md file with a table of scores, plus printed summary stats
(averages, win rate, latency comparison) - these are the real numbers that go
on the resume.
"""

import os
import time
import json
from anthropic import Anthropic
from basic_rag import answer_with_basic_rag, blunder_to_question
from multi_agent import answer_with_multi_agent
from game_analyzer import find_blunders
from dotenv import load_dotenv

load_dotenv()

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])


# Sample games to pull blunders from. Replace these with your own real games
# exported as PGN from chess.com or lichess for a more personal project.
# You need at least a few games to get enough blunders for a meaningful sample.
SAMPLE_GAMES = [
    """
[Event "Sample Game 1"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. b4 Bxb4 5. c3 Ba5 6. d4 exd4 7. O-O Qh4
8. Qb3 Nf6 9. e5 Ng4 10. Re1 d5 11. exd6 Bxc3 12. Nxc3 Qxc4 13. Qxc4 1-0
""",
    """
[Event "Sample Game 2"]
[White "Player3"]
[Black "Player4"]
[Result "0-1"]

1. d4 Nf6 2. c4 g6 3. Nc3 Bg7 4. e4 d6 5. Nf3 O-O 6. Be2 e5 7. O-O Nc6
8. d5 Ne7 9. Ne1 Nd7 10. Be3 f5 11. f3 f4 12. Bf2 g5 13. Nd3 Nf6
14. c5 Ng6 15. cxd6 cxd6 16. Rc1 Rf7 17. Nb5 Ne8 18. Qa4 a6 19. Nc3 Bd7
20. Qb3 b5 0-1
""",
    # Add more of your own games here as PGN strings for a richer dataset
]


def collect_blunders():
    """
    Runs Stockfish blunder detection across all sample games and combines
    the results into one list. Each blunder gets tagged with which game it
    came from for traceability.
    """
    all_blunders = []
    for game_idx, pgn in enumerate(SAMPLE_GAMES):
        print(f"Analyzing game {game_idx + 1}/{len(SAMPLE_GAMES)} for blunders...")
        blunders = find_blunders(pgn)
        for b in blunders:
            b["game_index"] = game_idx
        all_blunders.extend(blunders)
        print(f"  Found {len(blunders)} blunders")

    return all_blunders


def judge_responses(blunder_description, answer_a, answer_b):
    """
    Asks Claude to act as an impartial judge comparing two chess coaching
    explanations for the same blunder. Returns scores 1-10 for relevance and
    specificity for both, plus which one the judge preferred overall.
    """
    prompt = f"""You are an expert chess coach evaluator. You will see a description of a
mistake a player made and two different coaching explanations. Score each on a
scale of 1-10 for:

- RELEVANCE: Does the explanation directly address why this specific move was a mistake?
- SPECIFICITY: Does it give concrete, actionable advice rather than generic chess statements?

THE MISTAKE:
{blunder_description}

EXPLANATION A:
{answer_a}

EXPLANATION B:
{answer_b}

Respond ONLY with valid JSON in this exact format, no other text:
{{
  "answer_a_relevance": <1-10>,
  "answer_a_specificity": <1-10>,
  "answer_b_relevance": <1-10>,
  "answer_b_specificity": <1-10>,
  "preferred": "A" or "B",
  "reasoning": "<one sentence why>"
}}"""

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )

    raw_text = response.content[0].text.strip()
    raw_text = raw_text.replace("```json", "").replace("```", "").strip()

    return json.loads(raw_text)


def run_evaluation(blunders):
    """
    Main evaluation loop: for each detected blunder, runs both systems,
    judges the results, and collects everything into a results list.
    """
    results = []

    for idx, blunder in enumerate(blunders):
        description = blunder_to_question(blunder)
        print(f"\n[{idx + 1}/{len(blunders)}] Evaluating: {blunder['move']} (move {blunder['move_number']})")

        print("  Running System A (basic RAG)...")
        rag_result = answer_with_basic_rag(blunder)

        print("  Running System B (multi-agent)...")
        agent_result = answer_with_multi_agent(blunder)

        print("  Judging responses...")
        judgment = judge_responses(description, rag_result["answer"], agent_result["answer"])

        results.append({
            "blunder": f"{blunder['player']} played {blunder['move']} (move {blunder['move_number']})",
            "swing_centipawns": blunder["swing_centipawns"],
            "rag_latency": rag_result["latency_seconds"],
            "agent_latency": agent_result["latency_seconds"],
            "rag_relevance": judgment["answer_a_relevance"],
            "rag_specificity": judgment["answer_a_specificity"],
            "agent_relevance": judgment["answer_b_relevance"],
            "agent_specificity": judgment["answer_b_specificity"],
            "preferred": judgment["preferred"],
            "reasoning": judgment["reasoning"]
        })

        time.sleep(0.5)

    return results


def summarize_and_save(results):
    """
    Computes summary statistics and writes everything to results.md.
    These are the numbers you'll actually cite on your resume.
    """
    n = len(results)

    if n == 0:
        print("No blunders found in sample games - add more games or lower BLUNDER_THRESHOLD in game_analyzer.py")
        return

    avg_rag_relevance = sum(r["rag_relevance"] for r in results) / n
    avg_rag_specificity = sum(r["rag_specificity"] for r in results) / n
    avg_agent_relevance = sum(r["agent_relevance"] for r in results) / n
    avg_agent_specificity = sum(r["agent_specificity"] for r in results) / n

    avg_rag_latency = sum(r["rag_latency"] for r in results) / n
    avg_agent_latency = sum(r["agent_latency"] for r in results) / n

    agent_win_rate = sum(1 for r in results if r["preferred"] == "B") / n * 100

    rag_combined = avg_rag_relevance + avg_rag_specificity
    agent_combined = avg_agent_relevance + avg_agent_specificity
    pct_improvement = ((agent_combined - rag_combined) / rag_combined) * 100

    latency_multiplier = avg_agent_latency / avg_rag_latency

    summary = f"""# Chess Multi-Agent vs Basic RAG - Evaluation Results

Evaluated on {n} real blunders detected by Stockfish across {len(SAMPLE_GAMES)} sample games.

## Summary Statistics

| Metric | Basic RAG (System A) | Multi-Agent (System B) |
|---|---|---|
| Avg Relevance Score | {avg_rag_relevance:.1f}/10 | {avg_agent_relevance:.1f}/10 |
| Avg Specificity Score | {avg_rag_specificity:.1f}/10 | {avg_agent_specificity:.1f}/10 |
| Avg Latency | {avg_rag_latency:.2f}s | {avg_agent_latency:.2f}s |

**Multi-agent system was preferred in {agent_win_rate:.0f}% of blunders.**

**Combined quality score improvement: {pct_improvement:+.1f}%**

**Latency tradeoff: {latency_multiplier:.1f}x slower**

## Per-Blunder Results

| Blunder | Eval Swing | RAG (Rel/Spec) | Agent (Rel/Spec) | Preferred | Reasoning |
|---|---|---|---|---|---|
"""

    for r in results:
        summary += f"| {r['blunder']} | {r['swing_centipawns']}cp | {r['rag_relevance']}/{r['rag_specificity']} | {r['agent_relevance']}/{r['agent_specificity']} | {r['preferred']} | {r['reasoning']} |\n"

    with open("results.md", "w") as f:
        f.write(summary)

    print("\n" + "=" * 60)
    print("EVALUATION COMPLETE")
    print("=" * 60)
    print(f"Blunders analyzed: {n}")
    print(f"Multi-agent win rate: {agent_win_rate:.0f}%")
    print(f"Combined quality improvement: {pct_improvement:+.1f}%")
    print(f"Latency tradeoff: {latency_multiplier:.1f}x")
    print(f"\nFull results saved to results.md")


if __name__ == "__main__":
    print("Step 1: Detecting blunders in sample games using Stockfish...")
    blunders = collect_blunders()
    print(f"\nTotal blunders found: {len(blunders)}")

    if len(blunders) == 0:
        print("No blunders detected. Add more games to SAMPLE_GAMES or lower BLUNDER_THRESHOLD.")
        exit()

    print("\nStep 2: Running both systems on each blunder and judging...")
    results = run_evaluation(blunders)

    with open("raw_results.json", "w") as f:
        json.dump(results, f, indent=2)

    summarize_and_save(results)
