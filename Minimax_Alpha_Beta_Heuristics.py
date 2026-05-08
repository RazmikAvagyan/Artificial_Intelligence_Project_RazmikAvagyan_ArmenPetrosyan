import random
from Grid_And_Figures import check_win, is_full, empty_cells
from Heuristics import PieceSelectionHeuristic, PiecePlacementHeuristic

EXACT, LOWER, UPPER = 0, 1, 2
_tt = {}


def _bkey(board):
    return tuple(board[r][c] for r in range(4) for c in range(4))


def negamax_place(board, remaining, piece, depth, alpha, beta):
    orig_alpha = alpha
    key = (_bkey(board), frozenset(remaining), piece, depth)

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

    # move ordering: try cells that win immediately first
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

        if check_win(board):
            board[r][c] = None
            _tt[key] = (1000, EXACT)
            return (r, c), 1000

        if not remaining:
            board[r][c] = None
            if 0 > best_val:
                best_val = 0
                best_pos = (r, c)
            alpha = max(alpha, best_val)
            if alpha >= beta:
                break
            continue

        if depth <= 0:
            score = PiecePlacementHeuristic(r, c, piece, board, list(remaining))
            board[r][c] = None
        else:
            # same player's perspective after placing: pass alpha/beta unchanged
            _, score = negamax_choose(board, remaining, depth - 1, alpha, beta)
            board[r][c] = None

        if score > best_val:
            best_val = score
            best_pos = (r, c)

        alpha = max(alpha, best_val)
        if alpha >= beta:
            break

    if best_val <= orig_alpha:
        _tt[key] = (best_val, UPPER)
    elif best_val >= beta:
        _tt[key] = (best_val, LOWER)
    else:
        _tt[key] = (best_val, EXACT)

    return best_pos, best_val


def negamax_choose(board, remaining, depth, alpha, beta):
    orig_alpha = alpha
    key = (_bkey(board), frozenset(remaining), depth, 'c')

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

    if depth <= 0:
        scores = [(PieceSelectionHeuristic(p, board), p) for p in remaining]
        best_val = max(s for s, _ in scores)
        best_pieces = [p for s, p in scores if s == best_val]
        return random.choice(best_pieces), best_val

    # move ordering: most dangerous pieces for opponent first
    sorted_remaining = sorted(remaining, key=lambda p: PieceSelectionHeuristic(p, board))

    best_val = -9999
    best_piece = sorted_remaining[0]

    for piece in sorted_remaining:
        new_remaining = [p for p in remaining if p != piece]
        # perspective switches here: opponent places next, so negate the window
        _, val = negamax_place(board, new_remaining, piece, depth - 1, -beta, -alpha)
        val = -val

        if val > best_val:
            best_val = val
            best_piece = piece

        alpha = max(alpha, best_val)
        if alpha >= beta:
            break

    if best_val <= orig_alpha:
        _tt[key] = (best_val, UPPER)
    elif best_val >= beta:
        _tt[key] = (best_val, LOWER)
    else:
        _tt[key] = (best_val, EXACT)

    return best_piece, best_val


class MinimaxAgent:
    def __init__(self, depth):
        self.depth = depth

    def choose_piece(self, board, remaining, is_max):
        _tt.clear()
        piece, _ = negamax_choose(board, remaining, self.depth, -9999, 9999)
        if piece is None:
            piece = max(remaining, key=lambda p: PieceSelectionHeuristic(p, board))
        return piece

    def place_piece(self, board, remaining, piece, is_max):
        _tt.clear()
        pos, _ = negamax_place(board, remaining, piece, self.depth, -9999, 9999)
        if pos is None:
            pos = empty_cells(board)[0]
        return pos