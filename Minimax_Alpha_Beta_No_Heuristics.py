import random
from Grid_And_Figures import check_win, is_full, empty_cells

# Global transposition table for memoization
_tt = {}

def _bkey(board):
    """Generates a hashable tuple representation of the current board state."""
    return tuple(board[r][c] for r in range(4) for c in range(4))


def negamax_place(board, remaining, piece, depth, alpha, beta):
    """
    Evaluates and selects the optimal grid position to place the current piece.
    Implements Alpha-Beta pruning and transposition table lookups.
    """
    # Transposition table lookup
    key = (_bkey(board), frozenset(remaining), piece, depth)
    if key in _tt:
        return None, _tt[key]

    cells = empty_cells(board)

    # Immediate win condition check (evaluated independently of search depth)
    for r, c in cells:
        board[r][c] = piece
        won = check_win(board)
        board[r][c] = None
        if won:
            _tt[key] = 1000
            return (r, c), 1000

    # Terminal state check: Game over or draw
    if not cells or not remaining:
        _tt[key] = 0
        return cells[0] if cells else None, 0

    # Depth limit reached: Default to a neutral score
    if depth <= 0:
        _tt[key] = 0
        return cells[0], 0

    best_val = -9999
    best_pos = cells[0]

    for r, c in cells:
        board[r][c] = piece
        # Next action belongs to the same player (choosing a piece) - no score negation required
        _, val = negamax_choose(board, remaining, depth - 1, alpha, beta)
        board[r][c] = None  # Revert board state

        if val > best_val:
            best_val = val
            best_pos = (r, c)

        # Alpha-Beta pruning: Only performed on exact terminal win/loss values
        if best_val == 1000 or best_val == -1000:
            alpha = max(alpha, best_val)
            if alpha >= beta:
                break

    # Cache evaluation results before returning
    _tt[key] = best_val
    return best_pos, best_val


def negamax_choose(board, remaining, depth, alpha, beta):
    """
    Evaluates and selects the best piece to pass to the opponent.
    Implements Alpha-Beta pruning, transposition table lookups, and score negation.
    """
    # Transposition table lookup
    key = (_bkey(board), frozenset(remaining), depth, 'c')
    if key in _tt:
        return None, _tt[key]

    # Terminal state check
    if not remaining:
        _tt[key] = 0
        return None, 0

    # Depth limit reached: Fall back to a random choice with a neutral score
    if depth <= 0:
        piece = random.choice(remaining)
        return piece, 0

    best_val = -9999
    best_piece = remaining[0]

    for piece in remaining:
        new_remaining = [p for p in remaining if p != piece]
        # Opponent places the piece next: Negate search window boundaries and final value
        _, val = negamax_place(board, new_remaining, piece, depth - 1, -beta, -alpha)
        val = -val

        if val > best_val:
            best_val = val
            best_piece = piece

        # Alpha-Beta pruning: Only performed on exact terminal win/loss values
        if best_val == 1000 or best_val == -1000:
            alpha = max(alpha, best_val)
            if alpha >= beta:
                break

    # Cache evaluation results before returning
    _tt[key] = best_val
    return best_piece, best_val


class MinimaxAgentNoHeuristic:
    """
    Game agent utilizing depth-limited Negamax optimized with Alpha-Beta pruning
    and a clearing transposition table per turn.
    """
    def __init__(self, depth):
        self.depth = depth

    def choose_piece(self, board, remaining, is_max):
        """Determines the piece to pass to the opponent; clears the cache before evaluating."""
        _tt.clear()
        piece, _ = negamax_choose(board, remaining, self.depth, -1000, 1000)
        if piece is None:
            piece = random.choice(remaining)
        return piece

    def place_piece(self, board, remaining, piece, is_max):
        """Determines where to place the piece; clears the cache before evaluating."""
        _tt.clear()
        pos, _ = negamax_place(board, remaining, piece, self.depth, -1000, 1000)
        if pos is None:
            pos = empty_cells(board)[0]
        return pos