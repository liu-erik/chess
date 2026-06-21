"""
knowledge_base.py

This file does two things:
1. Defines our chess knowledge as a list of text "chunks" (like paragraphs from a chess book)
2. Embeds those chunks into ChromaDB so we can search them later using semantic similarity

Why chunks? RAG (Retrieval Augmented Generation) works by breaking knowledge into small
pieces, converting each piece into a vector (a list of numbers representing its meaning),
and storing those vectors. When a user asks a question, we convert the question into a
vector too, then find which chunks are "closest" in meaning to the question.
"""

import os
import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ---- THE KNOWLEDGE BASE ----
# Each entry is a (id, text, category) tuple. In a real production system this would
# come from a database or scraped source. For this project we hand-write it so we
# fully understand and control the content.

CHESS_CHUNKS = [
    # Opening principles
    ("open_1", "Control the center early. Pawns on e4, e5, d4, or d5 give your pieces more mobility and influence over the board.", "opening"),
    ("open_2", "Develop your knights before your bishops in most openings, since knight placement is usually more forced (knights to f3/c3 or f6/c6).", "opening"),
    ("open_3", "Castle early, ideally within the first 10 moves, to get your king to safety and connect your rooks.", "opening"),
    ("open_4", "Avoid moving the same piece twice in the opening unless necessary - every tempo matters when fighting for development.", "opening"),
    ("open_5", "Don't bring your queen out too early. It can be attacked by minor pieces, forcing you to waste moves retreating it.", "opening"),
    ("open_6", "The Italian Game (1.e4 e5 2.Nf3 Nc6 3.Bc4) targets the f7 weak point and develops quickly toward kingside attacks.", "opening"),
    ("open_7", "The Sicilian Defense (1.e4 c5) is the most popular response to e4 because it fights for the center asymmetrically and creates winning chances for Black.", "opening"),
    ("open_8", "The Queen's Gambit (1.d4 d5 2.c4) offers a pawn temporarily to gain central control and faster development.", "opening"),
    ("open_9", "In closed openings, prioritize piece maneuvering and pawn breaks over immediate tactical strikes.", "opening"),
    ("open_10", "Fianchetto setups (bishop on g2 or b2) provide long diagonal control but require careful pawn structure timing.", "opening"),

    # Tactics
    ("tac_1", "A fork is when one piece attacks two or more enemy pieces simultaneously. Knights are especially strong forking pieces due to their unique movement.", "tactics"),
    ("tac_2", "A pin restricts an enemy piece from moving because doing so would expose a more valuable piece behind it, often the king.", "tactics"),
    ("tac_3", "A skewer is similar to a pin but forces the more valuable piece to move first, exposing the piece behind it to capture.", "tactics"),
    ("tac_4", "A discovered attack happens when moving one piece reveals an attack from another piece that was previously blocked.", "tactics"),
    ("tac_5", "A double attack threatens two targets at once, forcing the opponent to lose material because they can only defend one.", "tactics"),
    ("tac_6", "Back rank weaknesses occur when a king has no escape squares because its own pawns block movement, making it vulnerable to rook or queen checks.", "tactics"),
    ("tac_7", "Removing the defender (also called deflection) involves capturing or attacking a piece that is the sole protector of another piece or square.", "tactics"),
    ("tac_8", "Zwischenzug, or 'in-between move,' is an unexpected intermediate move that changes the evaluation of a sequence before the expected recapture.", "tactics"),
    ("tac_9", "Overloading occurs when a single piece is responsible for defending too many things at once, allowing a tactic to exploit that overextension.", "tactics"),
    ("tac_10", "X-ray attacks occur when a piece attacks through another piece on the same file, rank, or diagonal, creating hidden pressure.", "tactics"),

    # Endgames
    ("end_1", "In king and pawn endgames, the concept of opposition (kings facing each other with one square between them) determines who controls key squares.", "endgame"),
    ("end_2", "The rule of the square helps determine if a king can catch a passed pawn before it promotes, without needing to calculate every move.", "endgame"),
    ("end_3", "Rook endgames are the most common endgame type. The general principle is to keep your rook active, even at the cost of a pawn.", "endgame"),
    ("end_4", "In king and pawn vs king endgames, the defending king must reach the key squares in front of the pawn to hold a draw.", "endgame"),
    ("end_5", "Two bishops working together (the 'bishop pair') are often stronger than a bishop and knight in open endgame positions.", "endgame"),
    ("end_6", "Lucena position is a winning technique in rook endgames where the attacking side builds a 'bridge' to shield their king from checks while promoting a pawn.", "endgame"),
    ("end_7", "Philidor position is a key defensive drawing technique in rook endgames, keeping the rook on the third rank to prevent the enemy king from advancing.", "endgame"),
    ("end_8", "Opposite colored bishop endgames are notoriously drawish even with extra pawns, because each bishop controls different colored squares.", "endgame"),
    ("end_9", "Knight endgames are technically pawn endgames in disguise - count tempo carefully since knights cannot lose a move like bishops can.", "endgame"),
    ("end_10", "Passed pawns should generally be pushed as far as possible, especially when the enemy king is far away (the 'outside passed pawn' principle).", "endgame"),

    # Strategy / positional concepts
    ("strat_1", "Weak squares are squares that cannot be defended by a pawn. Pieces, especially knights, become very strong when placed on weak squares.", "strategy"),
    ("strat_2", "An isolated pawn (no friendly pawns on adjacent files) can be a long-term weakness because it can never be defended by another pawn.", "strategy"),
    ("strat_3", "A pawn majority on one side of the board is an advantage that can be converted into a passed pawn in the endgame.", "strategy"),
    ("strat_4", "Open files should be controlled by rooks, since rooks gain mobility and attacking potential when no pawns block their path.", "strategy"),
    ("strat_5", "Doubled pawns (two pawns of the same color on the same file) are usually a structural weakness but can provide open lines for rooks.", "strategy"),
    ("strat_6", "Piece activity often matters more than material in the middlegame - a passive extra piece can be worse than an active one down material.", "strategy"),
    ("strat_7", "Prophylaxis means making a move that prevents your opponent's plan before they can execute it, rather than only pursuing your own plan.", "strategy"),
    ("strat_8", "Space advantage allows more piece mobility and restricts the opponent's pieces, but can become overextended if not supported.", "strategy"),
    ("strat_9", "The bishop pair is generally favored in open positions where both bishops have long, unobstructed diagonals.", "strategy"),
    ("strat_10", "Color complexes refer to weaknesses concentrated on one color of squares, often exploited by a single well-placed bishop.", "strategy"),

    # Common beginner mistakes
    ("mistake_1", "Moving too many pawns in the opening instead of developing pieces is a common beginner mistake that delays castling and piece activity.", "common_mistakes"),
    ("mistake_2", "Hanging pieces (leaving them undefended where they can be captured for free) is the most frequent way beginners lose material.", "common_mistakes"),
    ("mistake_3", "Not calculating checks, captures, and threats before every move leads to missing both opportunities and dangers.", "common_mistakes"),
    ("mistake_4", "Trading pieces without a clear reason can simplify into a worse endgame, especially when down material or space.", "common_mistakes"),
    ("mistake_5", "Chasing material gains while ignoring king safety often results in losing material back through a direct attack.", "common_mistakes"),
    ("mistake_6", "Failing to ask 'what is my opponent threatening' before making a move is one of the most common reasons tactics are missed.", "common_mistakes"),
    ("mistake_7", "Premature attacks without sufficient piece development usually fail and leave the attacker's own position weakened.", "common_mistakes"),
    ("mistake_8", "Ignoring pawn structure when trading pieces can create long-term weaknesses that are difficult to fix later in the game.", "common_mistakes"),
]


def get_openai_client():
    return OpenAI(api_key=os.environ["OPENAI_API_KEY"])


def embed_text(client, text):
    """
    Converts a piece of text into a vector (list of numbers) using OpenAI's
    embedding model. Similar meanings produce similar vectors, which is what
    lets us do semantic search later.
    """
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding


def build_knowledge_base():
    """
    Embeds all chess chunks and stores them in a local ChromaDB collection.
    This only needs to run once - ChromaDB persists the data to disk in ./chroma_db
    """
    client = get_openai_client()

    # Persistent client saves data to disk so we don't re-embed every run
    chroma_client = chromadb.PersistentClient(path="./chroma_db")

    # get_or_create_collection means: if it already exists, just use it.
    # This is what prevents the slow re-embedding-every-time problem.
    collection = chroma_client.get_or_create_collection(name="chess_knowledge")

    # If the collection already has data, skip re-embedding
    if collection.count() > 0:
        print(f"Knowledge base already exists with {collection.count()} chunks. Skipping embedding.")
        return collection

    print(f"Embedding {len(CHESS_CHUNKS)} chess chunks...")

    ids = []
    documents = []
    embeddings = []
    metadatas = []

    for idx, (chunk_id, text, category) in enumerate(CHESS_CHUNKS):
        print(f"  Embedding chunk {idx + 1}/{len(CHESS_CHUNKS)}: {chunk_id}")
        vector = embed_text(client, text)

        ids.append(chunk_id)
        documents.append(text)
        embeddings.append(vector)
        metadatas.append({"category": category})

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas
    )

    print(f"Done. Knowledge base has {collection.count()} chunks.")
    return collection


def get_collection():
    """Quick accessor to reuse the already-built collection without re-embedding."""
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    return chroma_client.get_or_create_collection(name="chess_knowledge")


if __name__ == "__main__":
    build_knowledge_base()
