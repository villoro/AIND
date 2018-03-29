"""This file is provided as a starting template for writing your own unit
tests to run and debug your minimax and alphabeta agents locally.  The test
cases used by the project assistant are not public.
"""

import unittest

import isolation
import game_agent

from importlib import reload


class IsolationTest(unittest.TestCase):
    """Unit tests for isolation agents"""

    def setUp(self):
        reload(game_agent)
        self.player1 = "Player1"
        self.player2 = "Player2"
        self.game = isolation.Board(self.player1, self.player2)


class MinimaxTest(unittest.TestCase):

	def test(self):

		self.game = isolation.Board("p1", "p2")

		self.game.apply_move((2, 3))
		self.game.apply_move((4, 2))
		self.game.apply_move((3, 5))
		self.game.apply_move((3, 0))

		player1 = game_agent.MinimaxPlayer()


class AlphaBetaTest(unittest.TestCase):

	pass


if __name__ == '__main__':
    unittest.main()
