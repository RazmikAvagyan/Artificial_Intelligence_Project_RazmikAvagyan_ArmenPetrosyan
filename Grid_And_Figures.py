import random

ALL_PIECES = [
    (a, b, c, d)
    for a in range(2)
    for b in range(2)
    for c in range(2)
    for d in range(2)
]

LINES = (
    [(0,0),(0,1),(0,2),(0,3)],
    [(1,0),(1,1),(1,2),(1,3)],
    [(2,0),(2,1),(2,2),(2,3)],
    [(3,0),(3,1),(3,2),(3,3)],
    [(0,0),(1,0),(2,0),(3,0)],
    [(0,1),(1,1),(2,1),(3,1)],
    [(0,2),(1,2),(2,2),(3,2)],
    [(0,3),(1,3),(2,3),(3,3)],
    [(0,0),(1,1),(2,2),(3,3)],
    [(0,3),(1,2),(2,1),(3,0)],
)

ATTR_CHARS = [('S','T'), ('L','D'), ('Q','C'), ('F','H')]

def piece_str(piece):
    return ''.join(ATTR_CHARS[i][piece[i]] for i in range(4))

def shares_attr(pieces):
    for attr in range(4):
        if len(set(p[attr] for p in pieces)) == 1:
            return True
    return False

def check_win(board):
    for line in LINES:
        cells = [board[r][c] for r, c in line]
        if None not in cells and shares_attr(cells):
            return True
    return False

def is_full(board):
    return all(board[r][c] is not None for r in range(4) for c in range(4))

def empty_cells(board):
    return [(r, c) for r in range(4) for c in range(4) if board[r][c] is None]

def print_board(board):
    print()
    print('        C0     C1     C2     C3')
    for r in range(4):
        row = f'  R{r} '
        for c in range(4):
            p = board[r][c]
            row += f' [{piece_str(p) if p else "    "}]'
        print(row)
    print()

def print_remaining(remaining):
    print('Available pieces:')
    for i, p in enumerate(remaining):
        print(f'  {i:2d}: {piece_str(p)}  {p}')
    print()

def generate_random_board_state(num_pieces: int):
    """
    Places `num_pieces` pieces randomly on the board.

    Returns:
        board     : 4x4 list with placed pieces (or None for empty cells)
        remaining : list of pieces not placed on the board

    Raises:
        ValueError: if num_pieces is not between 0 and 16.
    """
    if not (0 <= num_pieces <= 16):
        raise ValueError(f"num_pieces must be between 0 and 16, got {num_pieces}")

    board = [[None] * 4 for _ in range(4)]

    pieces = random.sample(ALL_PIECES, num_pieces)  # pick pieces without repetition
    cells = random.sample([(r, c) for r in range(4) for c in range(4)], num_pieces)

    for piece, (r, c) in zip(pieces, cells):
        board[r][c] = piece

    remaining = [p for p in ALL_PIECES if p not in pieces]

    if (not (check_win(board))):
        return board, remaining
    else:
        return None, None
