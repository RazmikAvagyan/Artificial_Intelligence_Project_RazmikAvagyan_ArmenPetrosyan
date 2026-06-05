from Grid_And_Figures import LINES, empty_cells, shares_attr, check_win


def PieceSelectionHeuristic(piece, board):
    """
    Evaluates the risk of giving a specific piece to the opponent.
    Returns the negative count of winning lines available to the opponent.
    """
    danger = 0
    for r, c in empty_cells(board):
        board[r][c] = piece
        for line in LINES:
            cells = [board[row][col] for row, col in line]
            # Check if placing this piece completes a winning line attribute match
            if None not in cells and shares_attr(cells):
                danger += 1
        board[r][c] = None  # Revert board state
    return -danger


def two_match_count(board, row, col):
    """
    Counts lines passing through (row, col) that contain exactly two pieces
    sharing at least one common attribute.
    """
    count = 0
    for line in LINES:
        if (row, col) not in line:
            continue
        cells = [board[r][c] for r, c in line]
        filled = [p for p in cells if p is not None]
        if len(filled) == 2:
            # Check for attribute alignment across the two pieces
            for attr in range(4):
                if len(set(p[attr] for p in filled)) == 1:
                    count += 1
                    break
    return count


def three_match_count(board, row, col):
    """
    Counts lines passing through (row, col) that contain exactly three pieces
    sharing at least one common attribute.
    """
    count = 0
    for line in LINES:
        if (row, col) not in line:
            continue
        cells = [board[r][c] for r, c in line]
        filled = [p for p in cells if p is not None]
        if len(filled) == 3:
            # Check for attribute alignment across the three pieces
            for attr in range(4):
                if len(set(p[attr] for p in filled)) == 1:
                    count += 1
                    break
    return count


def PiecePlacementHeuristic(row, col, piece, board, remaining):
    """
    Calculates the heuristic score for placing a piece at a specific position.
    Balances immediate wins, line setups, and the safety of remaining pieces.
    """
    board[row][col] = piece

    # Immediate win condition
    if check_win(board):
        board[row][col] = None
        return 1000

    # Calculate heuristic features
    f1 = two_match_count(board, row, col)
    f2 = three_match_count(board, row, col)

    # Evaluate opponent's best options from remaining pieces (minimax approach)
    worst_h1 = min((PieceSelectionHeuristic(p, board) for p in remaining), default=0)

    board[row][col] = None  # Revert board state

    # Weighted scoring formula
    w1, w2, w3 = 2, 5, 10
    return w1 * f1 + w2 * f2 + w3 * worst_h1