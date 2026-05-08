import random
import time
from Grid_And_Figures import ALL_PIECES, check_win, is_full, empty_cells
from Heuristics import PieceSelectionHeuristic, PiecePlacementHeuristic
from Minimax_Alpha_Beta_Heuristics import MinimaxAgent
from Minimax_Heuristics import MinimaxAgent as MinimaxAgent2
from Minimax_Alpha_Beta_No_Heuristics import MinimaxAgentNoHeuristic
from Minimax_No_Heuristics import MinimaxAgentNoHeuristic as MinimaxAgentNoHeuristic2
from Monte_Carlo import MCTSAgent


# ── MCTS Wrapper for Test Harness ─────────────────────────────────────────────

class MCTSTestWrapper:
    def __init__(self, iterations=100, piece_strategy="heuristic"):
        # player_id is updated dynamically in play_game
        # Pass the strategy through to the MCTSAgent
        self.agent = MCTSAgent(player_id=0, iterations=iterations, piece_strategy=piece_strategy)
        self.cached_piece = None

    def place_piece(self, board, remaining, piece, is_max):
        # Update internal player_id to match turn
        self.agent.player_id = 0 if is_max else 1
        # Run the MCTS simulation
        pos, next_piece = self.agent.place_and_choose(board, remaining, piece)
        # Store the piece choice for the subsequent choose_piece call
        self.cached_piece = next_piece
        return pos

    def choose_piece(self, board, remaining, is_max):
        return self.cached_piece


# ── Original Agents ───────────────────────────────────────────────────────────

class RandomAgent:
    def choose_piece(self, board, remaining, is_max):
        return random.choice(remaining)

    def place_piece(self, board, remaining, piece, is_max):
        return random.choice(empty_cells(board))


class HeuristicOnlyAgent:
    def choose_piece(self, board, remaining, is_max):
        return max(remaining, key=lambda p: PieceSelectionHeuristic(p, board))

    def place_piece(self, board, remaining, piece, is_max):
        best_pos, best_score = None, -99999
        for r, c in empty_cells(board):
            score = PiecePlacementHeuristic(r, c, piece, board, list(remaining))
            if score > best_score:
                best_score = score
                best_pos = (r, c)
        return best_pos


# ── Play one game ─────────────────────────────────────────────────────────────

def play_game(agent0, agent1):
    board = [[None] * 4 for _ in range(4)]
    remaining = list(ALL_PIECES)
    placer = 0

    piece = agent1.choose_piece(board, remaining, is_max=False)
    remaining.remove(piece)

    while True:
        current_agent = agent0 if placer == 0 else agent1

        pos = current_agent.place_piece(board, remaining, piece, is_max=(placer == 0))
        r, c = pos
        board[r][c] = piece

        if check_win(board):
            return placer
        if is_full(board):
            return None

        next_piece = current_agent.choose_piece(board, remaining, is_max=(placer == 0))
        if next_piece is not None:
            remaining.remove(next_piece)
            piece = next_piece

        placer = 1 - placer


# ── Run Matchup ───────────────────────────────────────────────────────────────

def run_matchup(name_a, agent_a, name_b, agent_b, n=50):
    wins_a = wins_b = draws = 0
    start_time = time.time()

    for i in range(1, n + 1):
        if i % 2 == 1:
            winner = play_game(agent_a, agent_b)
            if winner == 0:
                wins_a += 1
            elif winner == 1:
                wins_b += 1
            else:
                draws += 1
        else:
            winner = play_game(agent_b, agent_a)
            if winner == 0:
                wins_b += 1
            elif winner == 1:
                wins_a += 1
            else:
                draws += 1

    duration = time.time() - start_time
    print(f'{name_a:30s} vs {name_b:30s} | '
          f'{name_a}: {wins_a:3d}  {name_b}: {wins_b:3d}  Draws: {draws:3d}  ({n} games) '
          f'Time: {duration:.2f}s')


# ── Execution ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DEPTH = 3
    GAMES = 100
    Iteration1H = 15
    Iteration2H = 180
    Iteration3H = 2200
    Iteration4H = 110
    Iteration5H = 600
    Iteration1NoH = 5
    Iteration2NoH = 50
    Iteration3NoH = 600
    Iteration4NoH = 500
    Iteration5NoH = 8000
    Depth1 = 1
    Depth2 = 2
    Depth3 = 3
    Depth4 = 4

    random_agent = RandomAgent()
    heuristic_agent = HeuristicOnlyAgent()
    minimax_h = MinimaxAgent(depth=DEPTH)
    minimax_NoAB_H = MinimaxAgent2(depth=DEPTH)
    minimax_nh = MinimaxAgentNoHeuristic(depth=DEPTH)
    minimax_NoAB_NoH = MinimaxAgentNoHeuristic2(depth=DEPTH)

    minimax_hplus1 = MinimaxAgent(depth=DEPTH+1)
    minimax_nhplus1 = MinimaxAgentNoHeuristic(depth=DEPTH+1)


    # Instantiate MCTS with different strategies
    mcts_heuristic1 = MCTSTestWrapper(iterations=Iteration1H, piece_strategy="heuristic")
    mcts_random1 = MCTSTestWrapper(iterations=Iteration1NoH, piece_strategy="random")
    mcts_heuristic2 = MCTSTestWrapper(iterations=Iteration1H, piece_strategy="heuristic")
    mcts_random2 = MCTSTestWrapper(iterations=Iteration1NoH, piece_strategy="random")
    mcts_heuristic3 = MCTSTestWrapper(iterations=Iteration1H, piece_strategy="heuristic")
    mcts_random3 = MCTSTestWrapper(iterations=Iteration1NoH, piece_strategy="random")
    mcts_heuristic4 = MCTSTestWrapper(iterations=Iteration4H, piece_strategy="heuristic")
    mcts_random4 = MCTSTestWrapper(iterations=Iteration4NoH, piece_strategy="random")
    mcts_heuristic5 = MCTSTestWrapper(iterations=Iteration5H, piece_strategy="heuristic")
    mcts_random5 = MCTSTestWrapper(iterations=Iteration5NoH, piece_strategy="random")

    minimax_NoAB_H1 = MinimaxAgent2(depth = Depth1)
    minimax_NoAB_NoH1 = MinimaxAgentNoHeuristic2(depth = Depth1)
    minimax_NoAB_H2 = MinimaxAgent2(depth = Depth2)
    minimax_NoAB_NoH2 = MinimaxAgentNoHeuristic2(depth=Depth2)
    minimax_NoAB_H3 = MinimaxAgent2(depth = Depth3)
    minimax_NoAB_NoH3 = MinimaxAgentNoHeuristic2(depth = Depth3)
    minimax_AB_H1 = MinimaxAgent(depth=Depth3)
    minimax_AB_NoH1 = MinimaxAgentNoHeuristic(depth=Depth3)
    minimax_AB_H2 = MinimaxAgent(depth=Depth4)
    minimax_AB_NoH2 = MinimaxAgentNoHeuristic2(depth=Depth4)


    # print(f'Running matchups ({GAMES} games each, depth={DEPTH})\n')

    # Keep all your original commented matchups
    # run_matchup('Minimax + AB Heuristic', minimax_h, 'Minimax AB No Heuristic', minimax_nh, GAMES)
    # run_matchup('Minimax + No AB Heuristic', minimax_NoAB_H, 'Minimax No AB No Heuristic', minimax_NoAB_NoH, GAMES)
    # run_matchup('Minimax + AB Heuristic', minimax_h, 'Random', random_agent, GAMES)
    # run_matchup('Minimax + AB Heuristic', minimax_nh, 'Minimax + NO Ab Heuristic', minimax_NoAB_H, GAMES)
    # run_matchup('Minimax + No AB + No Heuristic', minimax_NoAB_NoH , 'Minimax + AB no Heuristics', minimax_nh, GAMES)

    # run_matchup('Minimax + AB Heuristic + 1', minimax_hplus1, 'Minimax + NO Ab Heuristic', minimax_NoAB_H, GAMES)
    # run_matchup('Minimax + No AB + No Heuristic', minimax_NoAB_NoH , 'Minimax + AB no Heuristics + 1', minimax_nhplus1, GAMES)
    # run_matchup('Minimax + AB Heuristic', minimax_h, 'Heuristic Only', heuristic_agent, GAMES)
    # run_matchup('Minimax + AB No Heuristic', minimax_nh, 'Heuristic Only', heuristic_agent, GAMES)
    # run_matchup('Minimax + No AB Heuristic', minimax_NoAB_H, 'Heuristic Only', heuristic_agent, GAMES)
    # run_matchup('Minimax + No AB No Heuristic', minimax_nh, 'Heuristic Only', heuristic_agent, GAMES)
    # run_matchup('Heuristic Only', heuristic_agent, 'Random', random_agent, GAMES)



    # Original active matchups
    # run_matchup('MCTS', mcts_agent, 'Minimax', minimax_NoAB_H, GAMES)
    # run_matchup('Minimax + Heuristic', minimax_NoAB_H, 'Random', random_agent, GAMES)
    # run_matchup('Minimax + Heuristic', minimax_NoAB_NoH, 'Random', random_agent, GAMES)

    # run_matchup('Minimax + AB Heuristic', minimax_AB_H1, 'Random', random_agent, GAMES)
    # run_matchup('Minimax + AB No Heuristic', minimax_AB_NoH1, 'Random', random_agent, GAMES)
    # run_matchup('mcts + Heuristic', mcts_heuristic4, 'Random', random_agent, GAMES)
    # run_matchup('mcts + Random', mcts_random4, 'Random', random_agent, GAMES)

    # run_matchup('Minimax + AB Heuristic', minimax_AB_H2, 'Random', random_agent, GAMES)
    # run_matchup('Minimax + AB No Heuristic', minimax_AB_NoH2, 'Random', random_agent, GAMES)
    # run_matchup('mcts + Heuristic', mcts_heuristic5, 'Random', random_agent, GAMES)
    # run_matchup('mcts + Random', mcts_random5, 'Random', random_agent, GAMES)

    # Try different MCTS strategies against Different Minimax
    run_matchup('MCTS (Heuristic) (iteration = 15)', mcts_heuristic1, 'Minimax No AB (Heuristic) (Depth = 1)', minimax_NoAB_H1, GAMES)
    run_matchup('MCTS (Heuristic) (iteration = 180)', mcts_heuristic2, 'Minimax No AB (Heuristic) (Depth = 2)', minimax_NoAB_H2, GAMES)
    run_matchup('MCTS (Heuristic Strategy) (iteration = 2200)', mcts_heuristic3, 'Minimax No AB (Heuristic Strategy) (Depth = 3)', minimax_NoAB_H3, GAMES)
    run_matchup('MCTS (Random) (iteration = 5)', mcts_random1, 'Minimax No AB (Random) (Depth = 1)', minimax_NoAB_NoH1, GAMES)
    run_matchup('MCTS (Random) (iteration = 50)', mcts_random2, 'Minimax No AB (Random) (Depth = 2)', minimax_NoAB_NoH2, GAMES)
    run_matchup('MCTS (Random)  (iteration = 600)', mcts_random3, 'Minimax No AB (Random) (Depth = 3)', minimax_NoAB_NoH3, GAMES)

    run_matchup('MCTS (Heuristic) (iteration = 110)', mcts_heuristic4, 'Minimax AB (Heuristic) (Depth = 3)', minimax_AB_H1, GAMES)
    run_matchup('MCTS (Heuristic Strategy) (iteration = 600)', mcts_heuristic5, 'Minimax No AB (Heuristic Strategy) (Depth = 4)', minimax_AB_H2, GAMES)
    run_matchup('MCTS (Random) (iteration = 500)', mcts_random4, 'Minimax No AB (Random) (Depth = 3)', minimax_AB_NoH1, GAMES)
    run_matchup('MCTS (Random)  (iteration = 8000)', mcts_random5, 'Minimax No AB (Random) (Depth = 4)', minimax_AB_NoH2, GAMES)