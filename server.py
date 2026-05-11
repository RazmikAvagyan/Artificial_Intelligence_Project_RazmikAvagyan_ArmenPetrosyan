"""
Quarto AI Server
────────────────
Run:   python server.py
Open:  http://localhost:5000      ← open THIS in your browser, not the .html file directly

Serving quarto.html through Flask means the page and the API share the same
origin (localhost:5000), so no CORS issues and no flask_cors needed.
"""

import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, send_from_directory

# ── Import your original modules ──────────────────────────────────────────────
from Grid_And_Figures import check_win, empty_cells
from Heuristics import PieceSelectionHeuristic, PiecePlacementHeuristic
from Minimax_Heuristics import MinimaxAgent, negamax_choose as mm_h_choose
from Minimax_No_Heuristics import MinimaxAgentNoHeuristic, negamax_choose as mm_nh_choose
from Minimax_Alpha_Beta_Heuristics import MinimaxAgent as ABMinimaxAgent, negamax_choose as ab_h_choose
from Minimax_Alpha_Beta_No_Heuristics import MinimaxAgentNoHeuristic as ABMinimaxAgentNoHeuristic, negamax_choose as ab_nh_choose
from Monte_Carlo import MCTSAgent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

# ── Serve the HTML ────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(BASE_DIR, "quarto.html")

# ── Conversion helpers ────────────────────────────────────────────────────────

def int_to_piece(n):
    return (n & 1, (n >> 1) & 1, (n >> 2) & 1, (n >> 3) & 1)

def piece_to_int(t):
    return t[0] + 2*t[1] + 4*t[2] + 8*t[3]

def flat_to_board(flat):
    board = [[None]*4 for _ in range(4)]
    for i, v in enumerate(flat):
        if v is not None:
            board[i // 4][i % 4] = int_to_piece(v)
    return board

def cell_to_flat(r, c):
    return r * 4 + c

# ── AI endpoint ───────────────────────────────────────────────────────────────

# iterations and depth now come from the request body (set in the HTML UI)

@app.route("/move", methods=["POST"])
def move():
    data     = request.get_json(force=True)
    algo     = data["algo"]
    board    = flat_to_board(data["board"])
    pool     = [int_to_piece(p) for p in data["pool"]]
    held_int = data.get("held")
    held     = int_to_piece(held_int) if held_int is not None else None
    phase    = data["phase"]
    pid      = int(data.get("player_id", 0))
    iterations = int(data.get("iterations", 1000))
    depth      = int(data.get("depth", 2))

    cell_flat  = None
    next_piece = None

    # ── SELECT phase ──────────────────────────────────────────────────────────
    if phase == "select":
        if algo == "random":
            chosen = random.choice(pool)

        elif algo == "heuristic":
            chosen = max(pool, key=lambda p: PieceSelectionHeuristic(p, board))

        elif algo == "minimax_nh":
            agent = MinimaxAgentNoHeuristic(depth=depth)
            chosen = agent.choose_piece(board, pool, is_max=True)
            if chosen is None:
                chosen = random.choice(pool)

        elif algo == "minimax_h":
            agent = MinimaxAgent(depth=depth)
            chosen = agent.choose_piece(board, pool, is_max=True)
            if chosen is None:
                chosen = max(pool, key=lambda p: PieceSelectionHeuristic(p, board))

        elif algo == "ab_nh":
            agent = ABMinimaxAgentNoHeuristic(depth=depth)
            chosen = agent.choose_piece(board, pool, is_max=True)
            if chosen is None:
                chosen = random.choice(pool)

        elif algo == "ab_h":
            agent = ABMinimaxAgent(depth=depth)
            chosen = agent.choose_piece(board, pool, is_max=True)
            if chosen is None:
                chosen = max(pool, key=lambda p: PieceSelectionHeuristic(p, board))

        elif algo in ("mcts_random", "mcts_heuristic"):
            strategy = "heuristic" if algo == "mcts_heuristic" else "random"
            agent = MCTSAgent(player_id=pid, iterations=iterations, piece_strategy=strategy)
            chosen = agent.choose_piece(board, pool)
            if chosen is None:
                chosen = random.choice(pool)

        else:
            chosen = random.choice(pool)

        next_piece = piece_to_int(chosen)

    # ── PLACE phase ───────────────────────────────────────────────────────────
    else:
        remaining = list(pool)   # pool already excludes held

        if algo == "random":
            cells = empty_cells(board)
            r, c  = random.choice(cells)
            cell_flat  = cell_to_flat(r, c)
            next_piece = piece_to_int(random.choice(remaining)) if remaining else None

        elif algo == "heuristic":
            cells = empty_cells(board)
            best_rc, best_score = cells[0], -9999
            for r, c in cells:
                s = PiecePlacementHeuristic(r, c, held, board, list(remaining))
                if s > best_score:
                    best_score = s
                    best_rc = (r, c)
            r, c = best_rc
            cell_flat = cell_to_flat(r, c)
            board[r][c] = held
            if remaining:
                chosen_next = max(remaining, key=lambda p: PieceSelectionHeuristic(p, board))
                next_piece  = piece_to_int(chosen_next)
            board[r][c] = None

        elif algo == "minimax_nh":
            agent = MinimaxAgentNoHeuristic(depth=depth)
            r, c  = agent.place_piece(board, remaining, held, is_max=True)
            cell_flat = cell_to_flat(r, c)
            board[r][c] = held
            chosen_next, _ = mm_nh_choose(board, remaining, depth=depth)
            next_piece = piece_to_int(chosen_next) if chosen_next is not None else (piece_to_int(random.choice(remaining)) if remaining else None)
            board[r][c] = None

        elif algo == "minimax_h":
            agent = MinimaxAgent(depth=depth)
            r, c  = agent.place_piece(board, remaining, held, is_max=True)
            cell_flat = cell_to_flat(r, c)
            board[r][c] = held
            chosen_next, _ = mm_h_choose(board, remaining, depth=depth)
            if chosen_next is None and remaining:
                chosen_next = max(remaining, key=lambda p: PieceSelectionHeuristic(p, board))
            next_piece = piece_to_int(chosen_next) if chosen_next is not None else None
            board[r][c] = None

        elif algo == "ab_nh":
            agent = ABMinimaxAgentNoHeuristic(depth=depth)
            r, c  = agent.place_piece(board, remaining, held, is_max=True)
            cell_flat = cell_to_flat(r, c)
            board[r][c] = held
            chosen_next, _ = ab_nh_choose(board, remaining, depth=depth, alpha=-1000, beta=1000)
            next_piece = piece_to_int(chosen_next) if chosen_next is not None else (piece_to_int(random.choice(remaining)) if remaining else None)
            board[r][c] = None

        elif algo == "ab_h":
            agent = ABMinimaxAgent(depth=depth)
            r, c  = agent.place_piece(board, remaining, held, is_max=True)
            cell_flat = cell_to_flat(r, c)
            board[r][c] = held
            chosen_next, _ = ab_h_choose(board, remaining, depth=depth, alpha=-9999, beta=9999)
            if chosen_next is None and remaining:
                chosen_next = max(remaining, key=lambda p: PieceSelectionHeuristic(p, board))
            next_piece = piece_to_int(chosen_next) if chosen_next is not None else None
            board[r][c] = None

        elif algo in ("mcts_random", "mcts_heuristic"):
            strategy = "heuristic" if algo == "mcts_heuristic" else "random"
            agent    = MCTSAgent(player_id=pid, iterations=iterations, piece_strategy=strategy)
            pos, chosen_next = agent.place_and_choose(board, remaining, held)
            r, c      = pos
            cell_flat = cell_to_flat(r, c)
            next_piece = piece_to_int(chosen_next) if chosen_next is not None else None

        else:
            cells = empty_cells(board)
            r, c  = random.choice(cells)
            cell_flat  = cell_to_flat(r, c)
            next_piece = piece_to_int(random.choice(remaining)) if remaining else None

    return jsonify({"cell": cell_flat, "next_piece": next_piece})


@app.route("/ping")
def ping():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("=" * 50)
    print("  Quarto AI server → http://localhost:5000")
    print("  Open that URL in your browser.")
    print("=" * 50)
    app.run(port=5000, debug=False)