# Chess Multi-Agent vs Basic RAG - Evaluation Results

Evaluated on 9 real blunders detected by Stockfish across 2 sample games.

## Summary Statistics

| Metric | Basic RAG (System A) | Multi-Agent (System B) |
|---|---|---|
| Avg Relevance Score | 7.0/10 | 8.8/10 |
| Avg Specificity Score | 5.8/10 | 8.9/10 |
| Avg Latency | 12.12s | 21.79s |

**Multi-agent system was preferred in 100% of blunders.**

**Combined quality score improvement: +38.3%**

**Latency tradeoff: 1.8x slower**

## Per-Blunder Results

| Blunder | Eval Swing | RAG (Rel/Spec) | Agent (Rel/Spec) | Preferred | Reasoning |
|---|---|---|---|---|---|
| Black played Qh4 (move 7) | 629cp | 7/6 | 9/9 | B | Explanation B provides a concrete hypothesis about what happened (g3 trapping the queen), gives a specific three-question decision framework, and uses more vivid language that makes the lesson memorable, while Explanation A remains more general about possible tactics. |
| White played Qb3 (move 8) | 509cp | 6/5 | 9/9 | B | Explanation B directly addresses the specific Qb3 mistake with concrete tactical patterns (queen trap, Na5, c5-c4) and actionable checklists, while Explanation A speaks more generally about opening principles without specific analysis of why Qb3 failed tactically. |
| Black played Nf6 (move 8) | 519cp | 8/6 | 9/9 | B | Explanation B provides a more actionable 5-second mental checklist with concrete steps to prevent future blunders, while A gives good analysis but less specific preventive methodology. |
| White played e5 (move 9) | 284cp | 7/6 | 8/9 | B | Explanation B provides a concrete three-question checklist tool that directly addresses when pawn advances are premature, while Explanation A gives more general advice about development priorities without actionable decision-making criteria. |
| Black played Ng4 (move 9) | 328cp | 8/7 | 9/9 | B | Explanation B provides a more actionable three-question framework for future decisions and more directly addresses why the knight placement itself (edge of board, few squares controlled) was problematic, while A focuses more generally on repeated piece moves. |
| White played Re1 (move 10) | 550cp | 7/6 | 9/8 | B | Explanation B directly addresses the specific context of being in a winning position (+6.63) and explains why Re1 was wrong in that particular situation (playing quietly when forcing moves were needed), while A treats it more generically as a tactical oversight without leveraging the massive advantage context. |
| Black played d5 (move 10) | 436cp | 7/6 | 8/9 | B | Explanation B provides a concrete three-step mental checklist and a practical drill for the next 5 games, making it significantly more actionable than Explanation A's more general bullet points. |
| Black played Ne8 (move 17) | 186cp | 7/5 | 9/9 | B | Explanation B directly addresses what the 1.9 pawn swing likely means (abandoned defense/allowed invasion) and provides concrete actionable steps like checking what squares are defended and looking for forward retreats with specific examples, while A stays more general about activity principles without explaining the dramatic evaluation change. |
| White played Qa4 (move 18) | 161cp | 6/5 | 9/9 | B | Explanation B directly addresses why Qa4 was a mistake with concrete tactical scenarios and actionable advice, while Explanation A spends most of its content disclaiming that it cannot properly address the mistake without seeing the position. |
