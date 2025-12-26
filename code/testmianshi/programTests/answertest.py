#Write your unit tests here

import math
import unittest
import sys

sys.path.append('/tmp/pycharm_project_481/code/testmianshi')

from extendmove.BishopMove import BishopMove
from extendmove.QueenMove import QueenMove
from chessLib.position import Position

class BishopTests(unittest.TestCase):
    def test_Bishop_move_from_inside_board(self):
        pos = Position(4, 4)
        Bishop = BishopMove()
        moves = Bishop.valid_moves(pos)
        self.assertIsNotNone(moves)
        self.assertEqual(4, moves.__len__())

        for move in moves:
            v = math.fabs(move.x - pos.x)
            if v == 1:
                self.assertEqual(1, math.fabs(move.y - pos.y))
            else:
                self.fail()

    def test_Bishop_move_from_corner(self):
        pos = Position(1, 1)
        Bishop = BishopMove()
        moves = Bishop.valid_moves(pos)
        self.assertIsNotNone(moves)
        self.assertEqual(1, moves.__len__())

        possibles = [Position(2, 2)]
        for move in possibles:
            self.assertTrue(moves.__contains__(move))

    def test_position(self):
        pos = Position(1, 1)
        self.assertEqual(1, pos.x)
        self.assertEqual(1, pos.y)
        pos2 = Position(1, 1)
        self.assertEqual(pos, pos2)

class QueenTests(unittest.TestCase):
    def test_Queen_move_from_inside_board(self):
        pos = Position(4, 4)
        Queen = QueenMove()
        moves = Queen.valid_moves(pos)
        self.assertIsNotNone(moves)
        self.assertEqual(8, moves.__len__())

        for move in moves:
            self.assertTrue(abs(move.x - pos.x) == 1 or abs(move.y - pos.y) == 1,
                            f"Invalid move detected: {move}")

    def test_Queen_move_from_corner(self):
        pos = Position(1, 1)
        Queen = QueenMove()
        moves = Queen.valid_moves(pos)
        self.assertIsNotNone(moves)
        self.assertEqual(3, moves.__len__())

        possibles = [Position(2, 2)]
        for move in possibles:
            self.assertTrue(moves.__contains__(move))

    def test_position(self):
        pos = Position(1, 1)
        self.assertEqual(1, pos.x)
        self.assertEqual(1, pos.y)
        pos2 = Position(1, 1)
        self.assertEqual(pos, pos2)
