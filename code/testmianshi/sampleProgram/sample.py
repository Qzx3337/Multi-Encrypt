from abc import ABC, abstractmethod
import sys
sys.path.append('/tmp/pycharm_project_481/code/testmianshi')
from chessLib.move import KnightMove
from chessLib.position import Position
import random



class BaseGame(ABC):
    @abstractmethod
    def play(self, moves: int):
        pass

    @abstractmethod
    def setup(self):
        pass


class SimpleGame(BaseGame):
    def __init__(self):
        self.__startPosition = None

    def play(self, moves: int):
        knight = KnightMove()
        pos = self.__startPosition
        print("0: My position is " + pos.to_string())

        for i in range(moves):
            possible_moves = knight.valid_moves(pos)
            r = random.randrange(0, possible_moves.__len__())
            pos = possible_moves[r]
            print(str(i) + ": My position is " + pos.to_string())

    def setup(self):
        self.__startPosition = Position(3, 3)
