import random
from Grid_And_Figures import check_win, is_full, empty_cells
from Heuristics import PieceSelectionHeuristic, PiecePlacementHeuristic


def negamax_place(board, remaining, piece, depth):
    cells = empty_cells(board)
    best_positions = []
    best_val = -9999

    for r, c in cells:
        board[r][c] = piece

        if check_win(board):
            board[r][c] = None
            return (r, c), 1000

        if is_full(board) or not remaining:
            board[r][c] = None
            return (r, c), 0

        if depth <= 0:
            score = PiecePlacementHeuristic(r, c, piece, board, list(remaining))
            if score > best_val:
                best_val = score
                best_positions = [(r, c)]
            elif score == best_val:
                best_positions.append((r, c))
            board[r][c] = None
            continue

        # same player places then chooses — no negation
        _, val = negamax_choose(board, remaining, depth - 1)
        if val > best_val:
            best_val = val
            best_positions = [(r, c)]
        elif val == best_val:
            best_positions.append((r, c))

        board[r][c] = None

    best_pos = random.choice(best_positions) if best_positions else None
    return best_pos, best_val


def negamax_choose(board, remaining, depth):
    if depth <= 0:
        scores = [(PieceSelectionHeuristic(p, board), p) for p in remaining]
        best_val = max(s for s, _ in scores)
        best_pieces = [p for s, p in scores if s == best_val]
        return random.choice(best_pieces), best_val

    best_pieces = []
    best_val = -9999

    for piece in remaining:
        new_remaining = [p for p in remaining if p != piece]
        # opponent places next — negate to convert to our perspective
        _, val = negamax_place(board, new_remaining, piece, depth - 1)
        val = -val

        if val > best_val:
            best_val = val
            best_pieces = [piece]
        elif val == best_val:
            best_pieces.append(piece)

    best_piece = random.choice(best_pieces) if best_pieces else None
    return best_piece, best_val


class MinimaxAgent:
    def __init__(self, depth):
        self.depth = depth

    def choose_piece(self, board, remaining, is_max):
        piece, _ = negamax_choose(board, remaining, self.depth)
        if piece is None:
            piece = max(remaining, key=lambda p: PieceSelectionHeuristic(p, board))
        return piece

    def place_piece(self, board, remaining, piece, is_max):
        pos, _ = negamax_place(board, remaining, piece, self.depth)
        if pos is None:
            pos = empty_cells(board)[0]
        return pos