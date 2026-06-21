"""
game_analyzer.py

This file handles the "real chess analysis" part of the project:
1. Parse a pasted PGN (a chess game in standard notation)
2. Replay the game move by move on a virtual board
3. Use Stockfish (a free, open-source chess engine) to evaluate each position
4. Flag moves where the evaluation drops significantly - these are blunders
   or mistakes worth coaching on

This replaces "guessing" what's a mistake with actual engine verification,
which is what makes the coaching feedback grounded in something real rather
than an LLM's opinion about a chess position.
"""

import chess
import chess.pgn
import chess.engine
import io

# Path to your Stockfish executable - update this after downloading Stockfish
# from https://stockfishchess.org/download/
# On Windows it'll be something like: "./stockfish/stockfish-windows-x86-64.exe"
# On Mac: "./stockfish/stockfish-macos-x86-64" (may need: chmod +x to make executable)
STOCKFISH_PATH = "./stockfish/stockfish-windows-x86-64-avx2.exe"

# How many centipawns (1/100th of a pawn) the evaluation must drop for a move
# to be flagged as a mistake. 100 = roughly losing a pawn's worth of advantage.
BLUNDER_THRESHOLD = 150


def parse_pgn(pgn_text):
    """
    Takes raw PGN text (what you'd paste from chess.com or lichess) and
    returns a python-chess Game object we can replay move by move.
    """
    pgn_io = io.StringIO(pgn_text)
    game = chess.pgn.read_game(pgn_io)
    if game is None:
        raise ValueError("Could not parse PGN - check the format")
    return game


def evaluate_position(board, engine, depth=12):
    """
    Asks Stockfish to evaluate the current board position.
    Returns the evaluation in centipawns from White's perspective
    (positive = White is better, negative = Black is better).

    depth=12 keeps analysis fast enough for a demo project - higher depth
    is more accurate but slower. Stockfish at depth 12 is still far stronger
    than a human, so it's plenty for catching real mistakes.
    """
    info = engine.analyse(board, chess.engine.Limit(depth=depth))
    score = info["score"].white()

    # Handle mate scores (e.g. "mate in 3") by converting to a large number
    if score.is_mate():
        mate_in = score.mate()
        return 10000 if mate_in > 0 else -10000

    return score.score()


def find_blunders(pgn_text):
    """
    The main analysis function: replays the entire game, evaluating each
    position before and after each move, and flags moves where the
    evaluation swings badly against the player who just moved.

    Returns a list of dicts, each describing one flagged mistake:
    move number, the move played, evaluation before/after, and whose move
    it was.
    """
    game = parse_pgn(pgn_text)
    board = game.board()

    blunders = []

    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as engine:
        eval_before = evaluate_position(board, engine)

        for move_number, move in enumerate(game.mainline_moves(), start=1):
            player_to_move = "White" if board.turn == chess.WHITE else "Black"
            move_san = board.san(move)  # Standard Algebraic Notation, e.g. "Nf3"

            board.push(move)
            eval_after = evaluate_position(board, engine)

            # Evaluation swing from the perspective of the player who just moved.
            # If White moved, we want to know if eval dropped for White.
            # If Black moved, the sign flips since eval is always from White's view.
            if player_to_move == "White":
                swing = eval_before - eval_after
            else:
                swing = eval_after - eval_before

            if swing >= BLUNDER_THRESHOLD:
                blunders.append({
                    "move_number": (move_number + 1) // 2,
                    "player": player_to_move,
                    "move": move_san,
                    "eval_before": eval_before,
                    "eval_after": eval_after,
                    "swing_centipawns": swing,
                    "fen_before_move": board.fen()  # board state for context
                })

            eval_before = eval_after

    return blunders


if __name__ == "__main__":
    # Quick test with a short sample game containing a known blunder
    # (Black hangs a piece with ...Qh4?? allowing a fork)
    sample_pgn = """
[Event "Test Game"]
[Site "?"]
[Date "2026.01.01"]
[Round "1"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bc4 Bc5 4. b4 Bxb4 5. c3 Ba5 6. d4 exd4 7. O-O Qh4
8. Qb3 Nf6 9. e5 Ng4 10. Re1 1-0
"""
    print("Analyzing sample game for blunders...")
    print(f"(Make sure STOCKFISH_PATH is correctly set to your Stockfish executable)\n")

    blunders = find_blunders(sample_pgn)

    if not blunders:
        print("No blunders found above threshold.")
    for b in blunders:
        print(f"Move {b['move_number']} ({b['player']}): {b['move']}")
        print(f"  Eval swing: {b['swing_centipawns']} centipawns")
        print(f"  Before: {b['eval_before']}, After: {b['eval_after']}\n")
