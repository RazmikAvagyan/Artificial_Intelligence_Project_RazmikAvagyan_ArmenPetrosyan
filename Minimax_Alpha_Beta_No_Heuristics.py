import random
from Grid_And_Figures import check_win, is_full, empty_cells

_tt = {}

def _bkey(board):
    return tuple(board[r][c] for r in range(4) for c in range(4))


def negamax_place(board, remaining, piece, depth, alpha, beta):
    key = (_bkey(board), frozenset(remaining), piece, depth)
    if key in _tt:
        return None, _tt[key]

    cells = empty_cells(board)

    # Immediate win — always check regardless of depth
    for r, c in cells:
        board[r][c] = piece
        won = check_win(board)
        board[r][c] = None
        if won:
            _tt[key] = 1000
            return (r, c), 1000

    if not cells or not remaining:
        _tt[key] = 0
        return cells[0] if cells else None, 0

    if depth <= 0:
        _tt[key] = 0
        return cells[0], 0

    best_val = -9999
    best_pos = cells[0]

    for r, c in cells:
        board[r][c] = piece
        # Same player chooses next — no perspective change, pass window unchanged
        _, val = negamax_choose(board, remaining, depth - 1, alpha, beta)
        board[r][c] = None

        if val > best_val:
            best_val = val
            best_pos = (r, c)

        # Prune only on exact win/loss — never on heuristic scores
        if best_val == 1000 or best_val == -1000:
            alpha = max(alpha, best_val)
            if alpha >= beta:
                break

    _tt[key] = best_val
    return best_pos, best_val


def negamax_choose(board, remaining, depth, alpha, beta):
    key = (_bkey(board), frozenset(remaining), depth, 'c')
    if key in _tt:
        return None, _tt[key]

    if not remaining:
        _tt[key] = 0
        return None, 0

    if depth <= 0:
        piece = random.choice(remaining)
        return piece, 0

    best_val = -9999
    best_piece = remaining[0]

    for piece in remaining:
        new_remaining = [p for p in remaining if p != piece]
        # Perspective switches — opponent places next, negate window and result
        _, val = negamax_place(board, new_remaining, piece, depth - 1, -beta, -alpha)
        val = -val

        if val > best_val:
            best_val = val
            best_piece = piece

        # Prune only on exact win/loss
        if best_val == 1000 or best_val == -1000:
            alpha = max(alpha, best_val)
            if alpha >= beta:
                break

    _tt[key] = best_val
    return best_piece, best_val


class MinimaxAgentNoHeuristic:
    def __init__(self, depth):
        self.depth = depth

    def choose_piece(self, board, remaining, is_max):
        _tt.clear()
        piece, _ = negamax_choose(board, remaining, self.depth, -1000, 1000)
        if piece is None:
            piece = random.choice(remaining)
        return piece

    def place_piece(self, board, remaining, piece, is_max):
        _tt.clear()
        pos, _ = negamax_place(board, remaining, piece, self.depth, -1000, 1000)
        if pos is None:
            pos = empty_cells(board)[0]
        return pos