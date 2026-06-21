"""
app.py

A minimal Streamlit interface to demo the project: paste in a real chess
game (PGN), the app detects blunders using Stockfish, then shows both
System A (basic RAG) and System B (multi-agent) coaching explanations
side by side for each detected mistake.

Run with: streamlit run app.py
"""

import streamlit as st
import os

st.set_page_config(page_title="Chess Multi-Agent Coach", layout="wide")

st.title("Multi-Agentic Chess Guide")
st.caption("Paste a game (PGN). Stockfish finds your blunders. Two AI systems explain them - compare which gives better coaching.")

tab1, tab2 = st.tabs(["Analyze a Game", "Benchmark Results"])

with tab1:
    pgn_input = st.text_area(
        "Paste your PGN here:",
        height=200,
        placeholder='[Event "Game"]\n[White "You"]\n[Black "Opponent"]\n[Result "1-0"]\n\n1. e4 e5 2. Nf3 Nc6 ...'
    )

    if st.button("Find Blunders & Get Coaching") and pgn_input:
        from game_analyzer import find_blunders
        from basic_rag import answer_with_basic_rag, blunder_to_question
        from multi_agent import answer_with_multi_agent

        with st.spinner("Running Stockfish analysis..."):
            try:
                blunders = find_blunders(pgn_input)
            except Exception as e:
                st.error(f"Could not analyze game: {e}")
                blunders = []

        if not blunders:
            st.info("No significant blunders detected in this game.")
        else:
            st.success(f"Found {len(blunders)} blunder(s)")

            for i, blunder in enumerate(blunders):
                st.divider()
                st.markdown(f"### Move {blunder['move_number']}: {blunder['player']} played **{blunder['move']}**")
                st.caption(f"Evaluation swing: {blunder['swing_centipawns']} centipawns")

                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("System A: Basic RAG")
                    with st.spinner("Analyzing..."):
                        rag_result = answer_with_basic_rag(blunder)
                    st.write(rag_result["answer"])
                    st.caption(f"Latency: {rag_result['latency_seconds']}s")

                with col2:
                    st.subheader("System B: Multi-Agent")
                    with st.spinner("Analyzing..."):
                        agent_result = answer_with_multi_agent(blunder)
                    st.write(agent_result["answer"])
                    with st.expander("See agent's mistake classification"):
                        st.write(agent_result["analysis"])
                    st.caption(f"Latency: {agent_result['latency_seconds']}s")

with tab2:
    st.subheader("Benchmark Results")
    if os.path.exists("results.md"):
        with open("results.md", "r") as f:
            st.markdown(f.read())
    else:
        st.info("No results yet. Run `python eval.py` first to generate the benchmark comparison.")
