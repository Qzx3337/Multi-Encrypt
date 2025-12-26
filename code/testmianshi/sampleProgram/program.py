import random

from sample import SimpleGame
from answer import ComplexGame

import sys

sys.path.append('/tmp/pycharm_project_481/code/testmianshi')

from chessLib.position import Position
from chessLib.move import KnightMove
from chessLib.BishopMove import BishopMove
from chessLib.QueenMove import QueenMove

if __name__ == '__main__':
    class ChessBoard:
        def __init__(self):
            self.board = [[None] * 8 for _ in range(8)]
            self.pieces = [KnightMove(), BishopMove(), QueenMove()]

        def place_piece(self, piece, pos):
            self.board[pos.x - 1][pos.y - 1] = piece

        def random_move(self):
            piece = random.choice(self.pieces)
            pos = random.choice(piece.valid_moves(Position(random.randint(1, 8), random.randint(1, 8))))
            self.place_piece(piece, pos)


    # 示例用法
    #game = SimpleGame()
    game = ComplexGame()
    game.setup()
    game.play(15)
