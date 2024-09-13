import torch
import chess
import chessboard
from NNUE import


def nnue_evaluate(board_representation):
    """Evaluates the board using the NNUE model."""
    with torch.no_grad():
        return nnue_model(board_representation)


def min_max(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        # Convert board to the NNUE input format
        board_representation = convert_board_to_nnue_input(board)
        return nnue_evaluate(board_representation)

    legal_moves = list(board.legal_moves)

    if maximizing_player:
        max_eval = float('-inf')
        for move in legal_moves:
            board.push(move)  # Make the move
            eval = min_max(board, depth - 1, alpha, beta, False)
            board.pop()  # Undo the move
            max_eval = max(max_eval, eval)
            alpha = max(alpha, eval)
            if beta <= alpha:
                break  # Beta cut-off
        return max_eval
    else:
        min_eval = float('inf')
        for move in legal_moves:
            board.push(move)
            eval = min_max(board, depth - 1, alpha, beta, True)
            board.pop()
            min_eval = min(min_eval, eval)
            beta = min(beta, eval)
            if beta <= alpha:
                break  # Alpha cut-off
        return min_eval