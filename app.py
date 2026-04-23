import time
import copy
from flask import Flask, render_template, jsonify, request

app = Flask(__name__)

#  Game Logic Helpers

EMPTY = ""
PLAYER_X = "X"  # Human
PLAYER_O = "O"  # AI


def check_winner(board):
    """Return 'X', 'O', 'draw', or None."""
    lines = [
        # Rows
        [board[0], board[1], board[2]],
        [board[3], board[4], board[5]],
        [board[6], board[7], board[8]],
        # Columns
        [board[0], board[3], board[6]],
        [board[1], board[4], board[7]],
        [board[2], board[5], board[8]],
        # Diagonals
        [board[0], board[4], board[8]],
        [board[2], board[4], board[6]],
    ]
    for line in lines:
        if line[0] == line[1] == line[2] and line[0] != EMPTY:
            return line[0]
    if all(cell != EMPTY for cell in board):
        return "draw"
    return None


def get_available_moves(board):
    """Return list of indices of empty cells."""
    return [i for i, cell in enumerate(board) if cell == EMPTY]


#  Minimax Algorithm (standard, no pruning)

class MinimaxSolver:
    """Standard Minimax without pruning."""

    def __init__(self):
        self.nodes_explored = 0

    def minimax(self, board, depth, is_maximizing):
        """
        Minimax recursive evaluation.
        Maximizing player = AI (O), Minimizing player = Human (X).
        """
        self.nodes_explored += 1

        result = check_winner(board)
        if result == PLAYER_O:
            return 10 - depth   # AI wins (prefer faster wins)
        elif result == PLAYER_X:
            return depth - 10   # Human wins (prefer slower losses)
        elif result == "draw":
            return 0

        if is_maximizing:
            best_score = float("-inf")
            for move in get_available_moves(board):
                board[move] = PLAYER_O
                score = self.minimax(board, depth + 1, False)
                board[move] = EMPTY
                best_score = max(best_score, score)
            return best_score
        else:
            best_score = float("inf")
            for move in get_available_moves(board):
                board[move] = PLAYER_X
                score = self.minimax(board, depth + 1, True)
                board[move] = EMPTY
                best_score = min(best_score, score)
            return best_score

    def find_best_move(self, board):
        """Evaluate every available move and pick the best one."""
        self.nodes_explored = 0
        best_score = float("-inf")
        best_move = None

        for move in get_available_moves(board):
            board[move] = PLAYER_O
            score = self.minimax(board, 0, False)
            board[move] = EMPTY
            if score > best_score:
                best_score = score
                best_move = move

        return best_move


#  Alpha-Beta Pruning Algorithm

class AlphaBetaSolver:
    """Minimax enhanced with Alpha-Beta Pruning."""

    def __init__(self):
        self.nodes_explored = 0

    def alphabeta(self, board, depth, alpha, beta, is_maximizing):
        """
        Minimax with alpha-beta pruning.
        Alpha = best guaranteed score for maximizer.
        Beta  = best guaranteed score for minimizer.
        """
        self.nodes_explored += 1

        result = check_winner(board)
        if result == PLAYER_O:
            return 10 - depth
        elif result == PLAYER_X:
            return depth - 10
        elif result == "draw":
            return 0

        if is_maximizing:
            best_score = float("-inf")
            for move in get_available_moves(board):
                board[move] = PLAYER_O
                score = self.alphabeta(board, depth + 1, alpha, beta, False)
                board[move] = EMPTY
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break  # β cut-off — prune remaining branches
            return best_score
        else:
            best_score = float("inf")
            for move in get_available_moves(board):
                board[move] = PLAYER_X
                score = self.alphabeta(board, depth + 1, alpha, beta, True)
                board[move] = EMPTY
                best_score = min(best_score, score)
                beta = min(beta, best_score)
                if beta <= alpha:
                    break  # α cut-off — prune remaining branches
            return best_score

    def find_best_move(self, board):
        """Evaluate every available move using alpha-beta pruning."""
        self.nodes_explored = 0
        best_score = float("-inf")
        best_move = None

        for move in get_available_moves(board):
            board[move] = PLAYER_O
            score = self.alphabeta(board, 0, float("-inf"), float("inf"), False)
            board[move] = EMPTY
            if score > best_score:
                best_score = score
                best_move = move

        return best_move


#  Flask Routes


@app.route("/")
def index():
    """Serve the main game page."""
    return render_template("index.html")


@app.route("/api/move", methods=["POST"])
def make_move():
    """
    Receive the current board state, run BOTH algorithms,
    and return the best move along with performance metrics.

    Expected JSON body:
        { "board": ["X", "", "O", "", "", "", "", "", ""] }
    """
    data = request.get_json()
    board = data.get("board", [""] * 9)

    # Check if game is already over
    winner = check_winner(board)
    if winner:
        return jsonify({"error": "Game is already over", "winner": winner})

    # ── Run Minimax 
    minimax_solver = MinimaxSolver()
    board_copy_mm = list(board)

    start_mm = time.perf_counter()
    move_mm = minimax_solver.find_best_move(board_copy_mm)
    end_mm = time.perf_counter()

    time_mm = (end_mm - start_mm) * 1000  # milliseconds
    nodes_mm = minimax_solver.nodes_explored

    #  Run Alpha-Beta 
    ab_solver = AlphaBetaSolver()
    board_copy_ab = list(board)

    start_ab = time.perf_counter()
    move_ab = ab_solver.find_best_move(board_copy_ab)
    end_ab = time.perf_counter()

    time_ab = (end_ab - start_ab) * 1000
    nodes_ab = ab_solver.nodes_explored

    # Both algorithms should agree on the best move (optimal play)
    best_move = move_ab  # Use alpha-beta result (identical result, faster)

    # Apply the move
    board[best_move] = PLAYER_O
    winner = check_winner(board)

    return jsonify({
        "move": best_move,
        "board": board,
        "winner": winner,
        "metrics": {
            "minimax": {
                "time_ms": round(time_mm, 4),
                "nodes_explored": nodes_mm,
            },
            "alpha_beta": {
                "time_ms": round(time_ab, 4),
                "nodes_explored": nodes_ab,
            },
            "speedup": round(time_mm / time_ab, 2) if time_ab > 0 else 0,
            "nodes_saved": nodes_mm - nodes_ab,
            "prune_percentage": round(
                ((nodes_mm - nodes_ab) / nodes_mm) * 100, 1
            ) if nodes_mm > 0 else 0,
        },
    })


@app.route("/api/reset", methods=["POST"])
def reset_game():
    """Reset the board."""
    return jsonify({"board": [""] * 9, "winner": None})



#  Entry Point


if __name__ == "__main__":
    print("\n[*] Tic-Tac-Toe AI Server")
    print("    Open http://127.0.0.1:5000 in your browser\n")
    app.run(debug=True, port=5000)
