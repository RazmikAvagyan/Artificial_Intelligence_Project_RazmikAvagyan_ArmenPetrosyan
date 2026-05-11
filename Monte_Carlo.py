"""
MCTS for Quarto — fully revised.

Algorithm structure:
  - Each MCTSNode represents a game state where some `player` is about to
    place a `piece` they were given.
  - After the placement, the same player chooses a `next_piece` to hand the
    opponent. So one "move" = (cell, next_piece) pair.
  - `wins` is always tracked from `root_player`'s perspective. UCB inverts
    exploitation at opponent nodes so they correctly minimize root's wins.

Bug fixes vs. original:
  1. `get_legal_moves` no longer offers `next_piece` choices that give the
     opponent an immediate win (was the main reason MCTS lost).
  2. `choose_rollout_piece` filters dangerous pieces in simulations too —
     critical for the win signal not being garbage.
  3. `choose_piece` (initial select-phase) now filters too, regardless of
     strategy. Previously fell through to `random.choice` for non-heuristic.
  4. `place_and_choose` no-children fallback was passing the wrong board to
     `choose_piece` (board before placement, not after).
  5. `get_legal_moves` early-exits on terminal states, so we don't waste
     work computing moves that are never used.
  6. When ALL pieces are dangerous (forced loss), we now pick the least
     dangerous one rather than any piece.
  7. `backpropagate` is now iterative — no recursion.
  8. UCB inversion is now expressed in terms of `self.player` vs `root_player`
     (clearer than the original `child.player != root_player` formulation,
     equivalent in behavior).
"""

import math
import random

from Grid_And_Figures import check_win, is_full, empty_cells
from Heuristics import PieceSelectionHeuristic


DRAW_SCORE = 0.5
EXPLORATION = 1.41


# ─────────────────────────── danger / safety helpers ──────────────────────────

def piece_danger(piece, board):
    """Number of empty cells where placing `piece` would immediately win."""
    danger = 0
    for r, c in empty_cells(board):
        board[r][c] = piece
        if check_win(board):
            danger += 1
        board[r][c] = None
    return danger


def safe_pieces(remaining, board):
    """
    Pieces that do NOT give the opponent an immediate win.

    If every remaining piece is dangerous (forced-loss situation), return the
    least dangerous ones — picking randomly among them is still better than
    handing over the most-winning piece.
    """
    safe = [p for p in remaining if piece_danger(p, board) == 0]
    if safe:
        return safe

    # Forced loss — minimize damage by giving the piece with the fewest wins
    dangers = [(piece_danger(p, board), p) for p in remaining]
    min_d = min(d for d, _ in dangers)
    return [p for d, p in dangers if d == min_d]


# ─────────────────────────────── MCTS Node ────────────────────────────────────

class MCTSNode:
    def __init__(
        self,
        board,
        remaining,
        piece,
        player,
        root_player,
        piece_strategy="heuristic",
        parent=None,
        move=None,
    ):
        self.board = [row[:] for row in board]
        self.remaining = list(remaining)
        self.piece = piece              # piece this node's player must place
        self.player = player            # who places `piece`
        self.root_player = root_player
        self.piece_strategy = piece_strategy
        self.parent = parent
        self.move = move                # (r, c, next_piece) that led here

        self.children = []
        self.wins = 0.0                 # always counted for root_player
        self.visits = 0
        self.untried = self.get_legal_moves()

    # --- move generation --------------------------------------------------------

    def get_legal_moves(self):
        # No piece to place, or game already decided → no moves
        if self.piece is None:
            return []
        if check_win(self.board) or is_full(self.board):
            return []

        cells = empty_cells(self.board)
        moves = []

        for r, c in cells:
            temp_board = [row[:] for row in self.board]
            temp_board[r][c] = self.piece

            # If this placement wins, no piece needs to be handed over
            if check_win(temp_board):
                moves.append((r, c, None))
                continue

            if not self.remaining:
                moves.append((r, c, None))
                continue

            # Filter out pieces the opponent can win with immediately
            candidates = safe_pieces(self.remaining, temp_board)

            if self.piece_strategy == "heuristic":
                next_piece = max(
                    candidates,
                    key=lambda p: PieceSelectionHeuristic(p, temp_board),
                )
                moves.append((r, c, next_piece))

            elif self.piece_strategy == "random":
                moves.append((r, c, random.choice(candidates)))

            else:  # "all" — full branching factor
                for next_piece in candidates:
                    moves.append((r, c, next_piece))

        random.shuffle(moves)
        return moves

    def is_terminal(self):
        return (
            self.piece is None
            or check_win(self.board)
            or is_full(self.board)
        )

    # --- tree operations --------------------------------------------------------

    def expand(self):
        r, c, next_piece = self.untried.pop()

        new_board = [row[:] for row in self.board]
        new_board[r][c] = self.piece

        if next_piece is not None:
            new_remaining = [p for p in self.remaining if p != next_piece]
        else:
            new_remaining = []

        child = MCTSNode(
            board=new_board,
            remaining=new_remaining,
            piece=next_piece,
            player=1 - self.player,
            root_player=self.root_player,
            piece_strategy=self.piece_strategy,
            parent=self,
            move=(r, c, next_piece),
        )
        self.children.append(child)
        return child

    def ucb_score(self, child):
        """
        Standard UCT, with exploitation flipped at opponent nodes.
        `wins` is from root_player's perspective everywhere, so:
          - if WE (self) are root_player, we want high win_rate
          - if WE (self) are the opponent, we want low win_rate
        """
        if child.visits == 0:
            return math.inf

        win_rate = child.wins / child.visits

        if self.player == self.root_player:
            exploitation = win_rate
        else:
            exploitation = 1 - win_rate

        exploration = EXPLORATION * math.sqrt(
            math.log(max(1, self.visits)) / child.visits
        )
        return exploitation + exploration

    def best_child(self):
        return max(self.children, key=self.ucb_score)

    # --- rollout helpers --------------------------------------------------------

    def choose_rollout_piece(self, board, remaining):
        """Pick piece to give opponent during simulation — filters out wins."""
        if not remaining:
            return None

        candidates = safe_pieces(remaining, board)

        if self.piece_strategy == "heuristic":
            return max(
                candidates,
                key=lambda p: PieceSelectionHeuristic(p, board),
            )
        return random.choice(candidates)

    def choose_rollout_placement(self, board, piece):
        """Place piece during simulation — takes immediate wins, else random."""
        cells = empty_cells(board)

        for r, c in cells:
            board[r][c] = piece
            if check_win(board):
                board[r][c] = None
                return r, c
            board[r][c] = None

        return random.choice(cells)

    # --- simulation -------------------------------------------------------------

    def simulate(self):
        """
        Random playout from this node. Returns the winning player's id,
        or None for a draw.
        """
        board = [row[:] for row in self.board]
        remaining = list(self.remaining)
        piece = self.piece
        player = self.player

        # If state is already won, the previous move (by the other player) won
        if check_win(board):
            return 1 - player

        while piece is not None:
            cells = empty_cells(board)
            if not cells:
                return None

            r, c = self.choose_rollout_placement(board, piece)
            board[r][c] = piece

            if check_win(board):
                return player

            if is_full(board) or not remaining:
                return None

            next_piece = self.choose_rollout_piece(board, remaining)
            remaining.remove(next_piece)

            piece = next_piece
            player = 1 - player

        return None

    # --- backprop ---------------------------------------------------------------

    def backpropagate(self, winner):
        node = self
        while node is not None:
            node.visits += 1
            if winner == node.root_player:
                node.wins += 1.0
            elif winner is None:
                node.wins += DRAW_SCORE
            node = node.parent


# ─────────────────────────────── MCTS Agent ───────────────────────────────────

class MCTSAgent:
    def __init__(self, player_id, iterations=500, piece_strategy="heuristic"):
        self.player_id = player_id
        self.iterations = iterations
        self.piece_strategy = piece_strategy

    # --- piece selection (start-of-game / select phase) -------------------------

    def choose_piece(self, board, remaining):
        """Pick initial piece to give opponent — always safety-filtered."""
        if not remaining:
            return None

        candidates = safe_pieces(remaining, board)

        if self.piece_strategy == "heuristic":
            return max(
                candidates,
                key=lambda p: PieceSelectionHeuristic(p, board),
            )
        return random.choice(candidates)

    # --- tactical shortcut ------------------------------------------------------

    def immediate_winning_move(self, board, remaining, piece):
        """If we can win this turn, just do it and pick a safe piece."""
        for r, c in empty_cells(board):
            temp_board = [row[:] for row in board]
            temp_board[r][c] = piece
            if check_win(temp_board):
                next_piece = self.choose_piece(temp_board, remaining)
                return (r, c), next_piece
        return None

    # --- main entry point -------------------------------------------------------

    def place_and_choose(self, board, remaining, piece):
        # Step 1: take the immediate win if it exists
        win_move = self.immediate_winning_move(board, remaining, piece)
        if win_move is not None:
            return win_move

        # Step 2: build root and run MCTS
        root = MCTSNode(
            board=board,
            remaining=remaining,
            piece=piece,
            player=self.player_id,
            root_player=self.player_id,
            piece_strategy=self.piece_strategy,
        )

        for _ in range(self.iterations):
            node = root

            # Selection — descend through fully-expanded non-terminal nodes
            while (
                not node.is_terminal()
                and not node.untried
                and node.children
            ):
                node = node.best_child()

            # Expansion — try one new move
            if node.untried and not node.is_terminal():
                node = node.expand()

            # Simulation — random playout from here
            winner = node.simulate()

            # Backpropagation — iterative, no recursion
            node.backpropagate(winner)

        # Step 3: fallback if root has no children at all
        if not root.children:
            cells = empty_cells(board)
            if not cells:
                return None, None
            pos = random.choice(cells)
            temp_board = [row[:] for row in board]
            temp_board[pos[0]][pos[1]] = piece
            next_piece = self.choose_piece(temp_board, remaining)
            return pos, next_piece

        # Step 4: pick child with best win rate (root.player is root_player,
        # so high win_rate = good for us)
        best = max(
            root.children,
            key=lambda c: c.wins / c.visits if c.visits > 0 else -1,
        )
        r, c, next_piece = best.move

        # Step 5: final safety net — guarantee we never hand over a winner
        if next_piece is not None:
            board_after = [row[:] for row in board]
            board_after[r][c] = piece
            if piece_danger(next_piece, board_after) > 0:
                next_piece = self.choose_piece(board_after, remaining)

        return (r, c), next_piece