from sample import BaseGame

from abc import ABC, abstractmethod
import sys
sys.path.append('/tmp/pycharm_project_481/code/testmianshi')
from chessLib.move import KnightMove
from chessLib.position import Position

from extendmove.BishopMove import BishopMove
from extendmove.QueenMove import QueenMove
import random


class ComplexGame(BaseGame):
    def __init__(self):
        self.__startPosition_k = None
        self.__startPosition_b = None
        self.__startPosition_q = None
        self.__pos_k = None
        self.__pos_b = None
        self.__pos_q = None

    def play(self, moves: int):
        knight = KnightMove()
        bishop = BishopMove()
        queen = QueenMove()

        pos_k = self.__pos_k
        pos_b = self.__pos_b
        pos_q = self.__pos_q

        print(f"0: Knight position is {pos_k.to_string()}")
        print(f"0: Bishop position is {pos_b.to_string()}")
        print(f"0: Queen position is {pos_q.to_string()}")

        for i in range(moves):
            # 获取所有可能的移动
            possible_moves_k = knight.valid_moves(pos_k)
            possible_moves_b = bishop.valid_moves(pos_b)
            possible_moves_q = queen.valid_moves(pos_q)

            # 随机选择移动
            if possible_moves_k:
                pos_k = random.choice(possible_moves_k)
            if possible_moves_b:
                pos_b = random.choice(possible_moves_b)
            if possible_moves_q:
                pos_q = random.choice(possible_moves_q)

            # 解决位置冲突
            positions = [pos_k, pos_b, pos_q]
            while len(positions) < 3:  # 如果有重复的位置
                if pos_k == pos_q:
                    possible_moves_k = [move for move in possible_moves_k if move != pos_q]
                    if possible_moves_k:
                        pos_k = random.choice(possible_moves_k)
                if pos_b == pos_q:
                    possible_moves_b = [move for move in possible_moves_b if move != pos_q]
                    if possible_moves_b:
                        pos_b = random.choice(possible_moves_b)
                if pos_k == pos_b:
                    possible_moves_k = [move for move in possible_moves_k if move != pos_b]
                    if possible_moves_k:
                        pos_k = random.choice(possible_moves_k)
                positions = [pos_k, pos_b, pos_q]

            # 打印每一步的结果
            print(f"{i + 1}: Knight position is {pos_k.to_string()}")
            print(f"{i + 1}: Bishop position is {pos_b.to_string()}")
            print(f"{i + 1}: Queen position is {pos_q.to_string()}")

            # 更新当前位置
            self.__pos_k = pos_k
            self.__pos_b = pos_b
            self.__pos_q = pos_q

    def setup(self):
        self.__startPosition_k = Position(2, 1)
        self.__startPosition_b = Position(3, 3)
        self.__startPosition_q = Position(4, 4)
        self.__pos_k = self.__startPosition_k
        self.__pos_b = self.__startPosition_b
        self.__pos_q = self.__startPosition_q

        # 可选：确保起始位置不冲突,也就是可以随机生成初始位置
        # positions = [self.__pos_k, self.__pos_b, self.__pos_q]
        # while len(set(positions)) < 3:
        #     self.__startPosition_k = Position(random.randint(1, 8), random.randint(1, 8))
        #     self.__startPosition_b = Position(random.randint(1, 8), random.randint(1, 8))
        #     self.__startPosition_q = Position(random.randint(1, 8), random.randint(1, 8))
        #     positions = [self.__startPosition_k, self.__startPosition_b, self.__startPosition_q]
        #
        # self.__pos_k = self.__startPosition_k
        # self.__pos_b = self.__startPosition_b
        # self.__pos_q = self.__startPosition_q



