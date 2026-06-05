import random
from Grid_And_Figures import check_win, is_full, empty_cells
from Heuristics import PieceSelectionHeuristic, PiecePlacementHeuristic


def negamax_place(board, remaining, piece, depth):
    """
    Evaluates and selects the best grid position to place the given piece.
    Uses a modified Negamax approach with depth limiting.
    """
    cells = empty_cells(board)
    best_positions = []
    best_val = -9999

    for r, c in cells:
        board[r][c] = piece

        # Terminal state check: Immediate win
        if check_win(board):
            board[r][c] = None
            return (r, c), 1000

        # Terminal state check: Draw
        if is_full(board) or not remaining:
            board[r][c] = None
            return (r, c), 0

        # Depth limit reached: Evaluate board using heuristics
        if depth <= 0:
            score = PiecePlacementHeuristic(r, c, piece, board, list(remaining))
            if score > best_val:
                best_val = score
                best_positions = [(r, c)]
            elif score == best_val:
                best_positions.append((r, c))
            board[r][c] = None
            continue

        # Same player transitions to choosing a piece for the opponent (no negation)
        _, val = negamax_choose(board, remaining, depth - 1)
        if val > best_val:
            best_val = val
            best_positions = [(r, c)]
        elif val == best_val:
            best_positions.append((r, c))

        board[r][c] = None  # Revert board state

    # Break ties randomly among optimal positions
    best_pos = random.choice(best_positions) if best_positions else None
    return best_pos, best_val


def negamax_choose(board, remaining, depth):
    """
    Evaluates and selects the best piece to pass to the opponent.
    Uses standard Negamax negation to invert the score from the opponent's perspective.
    """
    # If depth limit reached, then use piece selection heuristic directly
    if depth <= 0:
        scores = [(PieceSelectionHeuristic(p, board), p) for p in remaining]
        best_val = max(s for s, _ in scores)
        best_pieces = [p for s, p in scores if s == best_val]
        return random.choice(best_pieces), best_val

    best_pieces = []
    best_val = -9999

    for piece in remaining:
        new_remaining = [p for p in remaining if p != piece]

        # If opponent places the piece next, then negate value to align perspectives
        _, val = negamax_place(board, new_remaining, piece, depth - 1)
        val = -val

        if val > best_val:
            best_val = val
            best_pieces = [piece]
        elif val == best_val:
            best_pieces.append(piece)

    # Break ties randomly among optimal pieces
    best_piece = random.choice(best_pieces) if best_pieces else None
    return best_piece, best_val


class MinimaxAgent:
    """
    Game agent wrapper that interfaces with the Negamax search functions.
    """

    def __init__(self, depth):
        self.depth = depth

    def choose_piece(self, board, remaining, is_max):
        """Selects a piece to give to the opponent; falls back to heuristic on failure."""
        piece, _ = negamax_choose(board, remaining, self.depth)
        if piece is None:
            piece = max(remaining, key=lambda p: PieceSelectionHeuristic(p, board))
        return piece

    def place_piece(self, board, remaining, piece, is_max):
        """Selects a position to place the current piece; falls back to first empty cell on failure."""
        pos, _ = negamax_place(board, remaining, piece, self.depth)
        if pos is None:
            pos = empty_cells(board)[0]
        return pos