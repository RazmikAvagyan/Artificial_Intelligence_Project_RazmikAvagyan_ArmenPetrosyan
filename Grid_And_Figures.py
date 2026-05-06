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