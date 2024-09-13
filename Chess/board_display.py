import chess
from chessboard import display
from time import sleep
base_fen = chess.Board().fen()


def display_board(move_list: list[str]) -> None:
    board = chess.Board()
    disp = display.start(board.fen())
    while not display.check_for_quit():
        if move_list:
            board.push_san(move_list.pop(0))
            display.update(game_board=disp, fen=board.fen())
        sleep(1)
