from Grid_And_Figures import LINES, empty_cells, shares_attr


def PieceSelectionHeuristic(piece, board):
    danger = 0
    for r, c in empty_cells(board):
        board[r][c] = piece
        for line in LINES:
            cells = [board[row][col] for row, col in line]
            if None not in cells and shares_attr(cells):
                danger += 1
        board[r][c] = None
    return -danger


def two_match_count(board, row, col):
    count = 0
    for line in LINES:
        if (row, col) not in line:
            continue
        cells = [board[r][c] for r, c in line]
        filled = [p for p in cells if p is not None]
        if len(filled) == 2:
            for attr in range(4):
                if len(set(p[attr] for p in filled)) == 1:
                    count += 1
                    break
    return count


def three_match_count(board, row, col):
    count = 0
    for line in LINES:
        if (row, col) not in line:
            continue
        cells = [board[r][c] for r, c in line]
        filled = [p for p in cells if p is not None]
        if len(filled) == 3:
            for attr in range(4):
                if len(set(p[attr] for p in filled)) == 1:
                    count += 1
                    break
    return count


def PiecePlacementHeuristic(row, col, piece, board, remaining):
    board[row][col] = piece

    f1 = two_match_count(board, row, col)
    f2 = three_match_count(board, row, col)
    worst_h1 = min((PieceSelectionHeuristic(p, board) for p in remaining), default=0)

    board[row][col] = None

    w1, w2, w3 = 2, 5, 10
    return w1 * f1 + w2 * f2 + w3 * worst_h1