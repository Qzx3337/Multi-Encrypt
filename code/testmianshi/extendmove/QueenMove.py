import sys

sys.path.append('/tmp/pycharm_project_481/code/testmianshi')

from chessLib.BishopMove import BishopMove

from chessLib.position import Position

class QueenMove(BishopMove):
    __moves = [(1, 0), (-1, 0), (0, 1), (0, -1),(1, 1), (-1, 1), (1, -1), (-1, -1)]

    def valid_moves(self, pos: Position) -> list:
        result = []
        for m in self.__moves:
            p = Position(pos.x + m[0], pos.y + m[1])
            if 8 >= p.x > 0 and 8 >= p.y > 0:
                result.append(p)
        return result