# Quarto AI Game

This project is an AI-based implementation of the board game **Quarto**. It includes a browser interface and several AI agents, including Random, Heuristic, Minimax, Alpha-Beta Minimax, and Monte-Carlo Tree Search agents.

## Project Files

- `server.py` — runs the Flask server for the browser game.
- `quarto.html` — the game interface shown in the browser.
- `Grid_And_Figures.py` — contains the board logic and win-checking functions.
- `Heuristics.py` — contains heuristic scoring functions.
- `Minimax_Heuristics.py` — Minimax agent with heuristics.
- `Minimax_No_Heuristics.py` — Minimax agent without heuristics.
- `Minimax_Alpha_Beta_Heuristics.py` — Alpha-Beta Minimax agent with heuristics.
- `Minimax_Alpha_Beta_No_Heuristics.py` — Alpha-Beta Minimax agent without heuristics.
- `Monte_Carlo.py` — Monte-Carlo Tree Search agent.
- `Main.py` and `Test.py` — scripts for running AI-vs-AI experiments.

## Requirements

You need **Python 3** installed.

The browser game uses Flask, so install it before running the server:

```bash
pip install flask
```

If your system uses `python3` and `pip3`, use:

```bash
pip3 install flask
```

## How to Run the Game

1. Open a terminal or command prompt.

2. Go to the project folder:

```bash
cd Artificial_Intelligence_Project_RazmikAvagyan_ArmenPetrosyan_finall
```

3. Install Flask if it is not installed yet:

```bash
pip install flask
```

4. Start the server by typing in the IDE's terminal:

```bash
python server.py
```

On some systems, use:

```bash
python3 server.py
```

5. After the server starts, open this link in your browser:

```text
http://localhost:5000
```

Important: do **not** open `quarto.html` directly. The game should be opened through the Flask server so that the browser interface can communicate correctly with the AI backend.

## How to Play

After opening `http://localhost:5000`, use the web interface to play the game. The interface allows you to select different AI algorithms and settings such as search depth or MCTS iterations.
