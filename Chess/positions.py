import chess
base_fen = chess.Board().fen()
def pgn_to_fen(pgn):
    with open(pgn, 'r') as pgn_file:
