# Multi-Agentic Chess Guide

Multi-agent systems get used a lot without much evidence they outperform simpler approaches for the task at hand. So I built two versions of the same chess coaching assistant - basic and multi-agent rag. I then benchmarked them against each other on real detected blunders to test which model performed better.

## How it works

**Step 1 - Real blunder detection** (`game_analyzer.py`)
Paste in a PGN -> replay the game move by move on a virtual board -> Stockfish (a free open-source chess engine) evaluates every position -> flag moves where the evaluation drops significantly. This grounds the whole project in actual chess mistakes rather than guessing.

**System A - Basic RAG** (`basic_rag.py`)
Detected blunder -> convert to description -> embedd -> retrieve top 3 relevant coaching chunks -> single LLM call -> explanation

**System B - LangGraph Multi-Agent** (`multi_agent.py`)
Detected blunder -> Retrieval Agent (same retrieval as System A) -> Analysis Agent (classifies what TYPE of mistake it was - tactical, positional, or principle-based) -> Coach Agent (synthesizes retrieved knowledge + analysis into personalized advice tied to the exact position) -> explanation

Both systems pull from the same knowledge base - 48 hand-written chess knowledge chunks covering opening principles, tactics, endgames, strategy, and common mistakes, embedded with OpenAI's `text-embedding-3-small` and stored locally in pgVector.

## How I evaluated it

I ran Stockfish over several sample games to detect real blunders, then ran each one through both systems. To score the explanations I used Claude as an LLM judge - it doesn't know which system produced which answer (just labeled "Answer A" and "Answer B") so the scoring stays unbiased. It rates each explanation on:

- Relevance (1-10): does it actually address why this specific move was a mistake
- Specificity (1-10): is it concrete and actionable, or just generic chess advice

I also tracked latency for each system using wall-clock timing, since I wanted to see the actual tradeoff, not just assume multi-agent is "better" without a cost.

## Results

Check `results.md` for the full breakdown once you run the evaluation yourself.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env
# add your ANTHROPIC_API_KEY and OPENAI_API_KEY to .env
```

Most important part: also need Stockfish locally (free, no API key needed):
1. Download from https://stockfishchess.org/download/
2. Place the executable in a `stockfish/` folder in this project
3. Update `STOCKFISH_PATH` in `game_analyzer.py` to point to it

## Usage

```bash
# 1. Build the knowledge base (one-time, embeds 48 chunks)
python knowledge_base.py

# 2. Run the full benchmark evaluation on sample games
python eval.py

# 3. Launch the interactive demo - paste your own games
streamlit run app.py
```

## Stack

Python, LangGraph, pgVector, Stockfish, python-chess, Claude API (Anthropic), OpenAI embeddings, Streamlit

## Project structure

```
chess/
├── game_analyzer.py    # PGN parsing + Stockfish blunder detection
├── knowledge_base.py   # Chess content + pgVector embedding setup
├── basic_rag.py         # System A: single-shot RAG on detected blunders
├── multi_agent.py       # System B: LangGraph 3-agent pipeline
├── eval.py               # Benchmark runner + LLM-as-judge scoring
├── app.py                 # Streamlit demo UI - paste PGN, get coached
├── requirements.txt
├── .env.example
└── results.md            # Generated after running eval.py
```
