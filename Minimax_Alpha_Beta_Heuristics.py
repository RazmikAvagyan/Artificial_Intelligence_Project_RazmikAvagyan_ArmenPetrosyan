import random
from Grid_And_Figures import check_win, is_full, empty_cells
from Heuristics import PieceSelectionHeuristic, PiecePlacementHeuristic

# Transposition table node types for principal variation search / alpha-beta bounds
EXACT, LOWER, UPPER = 0, 1, 2
_tt = {}


def _bkey(board):
    """Generates a hashable tuple representation of the current board state."""
    return tuple(board[r][c] for r in range(4) for c in range(4))


def negamax_place(board, remaining, piece, depth, alpha, beta):
    """
    Evaluates and selects the optimal grid position to place the current piece.
    Implements move ordering, heuristic fallback, and transposition table bound matching.
    """
    orig_alpha = alpha
    key = (_bkey(board), frozenset(remaining), piece, depth)

    # Transposition table lookup: adjust alpha/beta or return early on cutoff
    if key in _tt:
        val, flag = _tt[key]
        if flag == EXACT:
            return None, val
        if flag == LOWER:
            alpha = max(alpha, val)
        elif flag == UPPER:
            beta = min(beta, val)
        if alpha >= beta:
            return None, val

    cells = empty_cells(board)
    if not cells:
        return None, 0

    # Move ordering: prioritize cells that yield an immediate victory
    def win_first(rc):
        r, c = rc
        board[r][c] = piece
        w = check_win(board)
        board[r][c] = None
        return 1 if w else 0

    cells.sort(key=win_first, reverse=True)

    best_val = -9999
    best_pos = cells[0]

    for r, c in cells:
        board[r][c] = piece

        # Terminal state check: Immediate win
        if check_win(board):
            board[r][c] = None
            _tt[key] = (1000, EXACT)
            return (r, c), 1000

        # Terminal state check: End of game / draw
        if not remaining:
            board[r][c] = None
            if 0 > best_val:
                best_val = 0
                best_pos = (r, c)
            alpha = max(alpha, best_val)
            if alpha >= beta:
                break
            continue

        # Evaluate position via heuristic or by proceeding down the search tree
        if depth <= 0:
            score = PiecePlacementHeuristic(r, c, piece, board, list(remaining))
            board[r][c] = None
        else:
            # Same-player turn transition: Pass alpha/beta bounds unchanged
            _, score = negamax_choose(board, remaining, depth - 1, alpha, beta)
            board[r][c] = None

        if score > best_val:
            best_val = score
            best_pos = (r, c)

        # Alpha-Beta cutoff update
        alpha = max(alpha, best_val)
        if alpha >= beta:
            break

    # Cache the evaluation entry alongside its appropriate transposition bounds flag
    if best_val <= orig_alpha:
        _tt[key] = (best_val, UPPER)
    elif best_val >= beta:
        _tt[key] = (best_val, LOWER)
    else:
        _tt[key] = (best_val, EXACT)

    return best_pos, best_val


def negamax_choose(board, remaining, depth, alpha, beta):
    """
    Evaluates and selects the optimal piece to pass to the opponent.
    Implements move ordering based on danger heuristics and transposition logging.
    """
    orig_alpha = alpha
    key = (_bkey(board), frozenset(remaining), depth, 'c')

    # Transposition table lookup: adjust alpha/beta or return early on cutoff
    if key in _tt:
        val, flag = _tt[key]
        if flag == EXACT:
            return None, val
        if flag == LOWER:
            alpha = max(alpha, val)
        elif flag == UPPER:
            beta = min(beta, val)
        if alpha >= beta:
            return None, val

    if not remaining:
        return None, 0

    # Depth limit reached: Evaluate remaining options via selection heuristic
    if depth <= 0:
        scores = [(PieceSelectionHeuristic(p, board), p) for p in remaining]
        best_val = max(s for s, _ in scores)
        best_pieces = [p for s, p in scores if s == best_val]
        return random.choice(best_pieces), best_val

    # Move ordering: Sort pieces to evaluate the most dangerous choices for the opponent first
    sorted_remaining = sorted(remaining, key=lambda p: PieceSelectionHeuristic(p, board))

    best_val = -9999
    best_piece = sorted_remaining[0]

    for piece in sorted_remaining:
        new_remaining = [p for p in remaining if p != piece]
        # Opponent turn transition: Invert perspective by negating window and evaluation score
        _, val = negamax_place(board, new_remaining, piece, depth - 1, -beta, -alpha)
        val = -val

        if val > best_val:
            best_val = val
            best_piece = piece

        # Alpha-Beta cutoff update
        alpha = max(alpha, best_val)
        if alpha >= beta:
            break

    # Cache the evaluation entry alongside its appropriate transposition bounds flag
    if best_val <= orig_alpha:
        _tt[key] = (best_val, UPPER)
    elif best_val >= beta:
        _tt[key] = (best_val, LOWER)
    else:
        _tt[key] = (best_val, EXACT)

    return best_piece, best_val


class MinimaxAgent:
    """
    Heuristic-driven game agent interface optimized with move ordering,
    Alpha-Beta pruning, and full transposition table memoization.
    """
    def __init__(self, depth):
        self.depth = depth

    def choose_piece(self, board, remaining, is_max):
        """Selects a piece to hand over to the opponent; flushes the cache before evaluation."""
        _tt.clear()
        piece, _ = negamax_choose(board, remaining, self.depth, -9999, 9999)
        if piece is None:
            piece = max(remaining, key=lambda p: PieceSelectionHeuristic(p, board))
        return piece

    def place_piece(self, board, remaining, piece, is_max):
        """Selects a grid position to place the current piece; flushes the cache before evaluation."""
        _tt.clear()
        pos, _ = negamax_place(board, remaining, piece, self.depth, -9999, 9999)
        if pos is None:
            pos = empty_cells(board)[0]
        return pos