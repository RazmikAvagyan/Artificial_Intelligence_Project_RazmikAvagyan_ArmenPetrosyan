import math
import random

from Grid_And_Figures import check_win, is_full, empty_cells
from Heuristics import PieceSelectionHeuristic


DRAW_SCORE = 0.5


class MCTSNode:
    def __init__(
        self,
        board,
        remaining,
        piece,
        player,
        root_player,
        piece_strategy="heuristic",
        parent=None,
        move=None
    ):
        self.board = [row[:] for row in board]
        self.remaining = list(remaining)
        self.piece = piece
        self.player = player
        self.root_player = root_player
        self.piece_strategy = piece_strategy
        self.parent = parent
        self.move = move

        self.children = []
        self.wins = 0.0
        self.visits = 0
        self.untried = self.get_legal_moves()

    def get_legal_moves(self):
        if self.piece is None:
            return []

        cells = empty_cells(self.board)
        moves = []

        for r, c in cells:
            temp_board = [row[:] for row in self.board]
            temp_board[r][c] = self.piece

            if not self.remaining:
                moves.append((r, c, None))
            else:
                if self.piece_strategy == "heuristic":
                    next_piece = max(
                        self.remaining,
                        key=lambda p: PieceSelectionHeuristic(p, temp_board)
                    )
                    moves.append((r, c, next_piece))

                elif self.piece_strategy == "random":
                    next_piece = random.choice(self.remaining)
                    moves.append((r, c, next_piece))

                else:
                    for next_piece in self.remaining:
                        moves.append((r, c, next_piece))

        random.shuffle(moves)
        return moves

    def is_terminal(self):
        return check_win(self.board) or is_full(self.board) or self.piece is None

    def expand(self):
        r, c, next_piece = self.untried.pop()

        new_board = [row[:] for row in self.board]
        new_board[r][c] = self.piece

        if next_piece is not None:
            new_remaining = [p for p in self.remaining if p != next_piece]
        else:
            new_remaining = []

        child = MCTSNode(
            board=new_board,
            remaining=new_remaining,
            piece=next_piece,
            player=1 - self.player,
            root_player=self.root_player,
            piece_strategy=self.piece_strategy,
            parent=self,
            move=(r, c, next_piece)
        )

        self.children.append(child)
        return child

    def ucb_score(self, child, exploration=1.41):
        if child.visits == 0:
            return math.inf

        win_rate = child.wins / child.visits

        # If it is opponent's turn in the child node,
        # opponent will try to reduce root player's win rate.
        if child.player != self.root_player:
            exploitation = win_rate
        else:
            exploitation = 1 - win_rate

        exploration_term = exploration * math.sqrt(
            math.log(max(1, self.visits)) / child.visits
        )

        return exploitation + exploration_term

    def best_child(self):
        return max(self.children, key=lambda child: self.ucb_score(child))

    def choose_rollout_piece(self, board, remaining):
        if not remaining:
            return None

        if self.piece_strategy == "heuristic":
            return max(
                remaining,
                key=lambda p: PieceSelectionHeuristic(p, board)
            )

        return random.choice(remaining)

    def choose_rollout_placement(self, board, piece):
        cells = empty_cells(board)

        # First, if there is an immediate winning placement, use it.
        for r, c in cells:
            board[r][c] = piece
            if check_win(board):
                board[r][c] = None
                return r, c
            board[r][c] = None

        # Otherwise rollout placement is random.
        return random.choice(cells)

    def simulate(self):
        board = [row[:] for row in self.board]
        remaining = list(self.remaining)
        piece = self.piece
        player = self.player

        # If this board is already winning,
        # the previous player made the winning move.
        if check_win(board):
            return 1 - player

        while piece is not None:
            cells = empty_cells(board)

            if not cells:
                return None

            r, c = self.choose_rollout_placement(board, piece)
            board[r][c] = piece

            if check_win(board):
                return player

            if is_full(board) or not remaining:
                return None

            next_piece = self.choose_rollout_piece(board, remaining)
            remaining.remove(next_piece)

            piece = next_piece
            player = 1 - player

        return None

    def backpropagate(self, winner):
        self.visits += 1

        if winner == self.root_player:
            self.wins += 1.0
        elif winner is None:
            self.wins += DRAW_SCORE

        if self.parent is not None:
            self.parent.backpropagate(winner)


class MCTSAgent:
    def __init__(self, player_id, iterations=500, piece_strategy="heuristic"):
        self.player_id = player_id
        self.iterations = iterations
        self.piece_strategy = piece_strategy

    def choose_piece(self, board, remaining):
        if not remaining:
            return None

        if self.piece_strategy == "heuristic":
            return max(
                remaining,
                key=lambda p: PieceSelectionHeuristic(p, board)
            )

        return random.choice(remaining)

    def immediate_winning_move(self, board, remaining, piece):
        for r, c in empty_cells(board):
            temp_board = [row[:] for row in board]
            temp_board[r][c] = piece

            if check_win(temp_board):
                next_piece = self.choose_piece(temp_board, remaining)
                return (r, c), next_piece

        return None

    def place_and_choose(self, board, remaining, piece):
        # Step 1: if MCTS can win immediately, do it.
        win_move = self.immediate_winning_move(board, remaining, piece)

        if win_move is not None:
            return win_move

        # Step 2: create root node.
        root = MCTSNode(
            board=board,
            remaining=remaining,
            piece=piece,
            player=self.player_id,
            root_player=self.player_id,
            piece_strategy=self.piece_strategy
        )

        # Step 3: run MCTS iterations.
        for _ in range(self.iterations):
            node = root

            # Selection
            while (
                not node.is_terminal()
                and not node.untried
                and node.children
            ):
                node = node.best_child()

            # Expansion
            if node.untried and not node.is_terminal():
                node = node.expand()

            # Simulation
            winner = node.simulate()

            # Backpropagation
            node.backpropagate(winner)

        # Step 4: fallback if no children exist.
        if not root.children:
            cells = empty_cells(board)
            pos = random.choice(cells)
            next_piece = self.choose_piece(board, remaining)
            return pos, next_piece

        # Step 5: choose the child with best win rate.
        best_child = max(
            root.children,
            key=lambda child: child.wins / child.visits if child.visits > 0 else -1
        )

        r, c, next_piece = best_child.move
        return (r, c), next_piece