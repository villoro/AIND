"""Finish all TODO items in this file to complete the isolation project, then
    test your agent's strength against a set of known agents using tournament.py
    and include the results in your report.
"""
import random


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
        of the given player.

        This should be the best heuristic function for your project submission.

        Note: this function should be called from within a Player instance as
        `self.score()` -- you should not need to call this function directly.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        player : object
            A player instance in the current game (i.e., an object corresponding to
            one of the player objects `game.__player_1__` or `game.__player_2__`.)

        Returns
        -------
        float
            The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    game_percent = game.move_count/(game.width*game.height)

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    w, h = game.width / 2., game.height / 2.
    y, x = game.get_player_location(player)

    if game_percent < 0.75:
        return  10*float(own_moves - opp_moves) + float((h - y)**2 + (w - x)**2)

    corners = [(0, 0), (0, (game.width - 1)),
               ((game.height - 1), 0), ((game.height - 1), (game.width - 1))]

    own_mov_in_corners = [m for m in game.get_legal_moves(player) if m in corners]
    opp_mov_in_corners = [m for m in game.get_legal_moves(game.get_opponent(player)) if m in corners]

    return 10*float(own_moves - opp_moves) + \
                float((h - y)**2 + (w - x)**2) + \
                (len(own_mov_in_corners) - len(opp_mov_in_corners))


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
        of the given player.

        Note: this function should be called from within a Player instance as
        `self.score()` -- you should not need to call this function directly.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        player : object
            A player instance in the current game (i.e., an object corresponding to
            one of the player objects `game.__player_1__` or `game.__player_2__`.)

        Returns
        -------
        float
            The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    game_percent = game.move_count/(game.width*game.height)

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    w, h = game.width / 2., game.height / 2.
    y, x = game.get_player_location(player)

    if game_percent < 0.5:
        return 10*float(own_moves - opp_moves) + float((h - y)**2 + (w - x)**2)

    return float(own_moves - opp_moves)


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
        of the given player.

        Note: this function should be called from within a Player instance as
        `self.score()` -- you should not need to call this function directly.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        player : object
            A player instance in the current game (i.e., an object corresponding to
            one of the player objects `game.__player_1__` or `game.__player_2__`.)

        Returns
        -------
        float
            The heuristic value of the current game state to the specified player.
    """

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    game_percent = game.move_count/(game.width*game.height)

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    if game_percent < 0.5:
        return float(own_moves - opp_moves)

    return float(own_moves - 0.5*opp_moves)


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
        constructed or tested directly.

        ********************  DO NOT MODIFY THIS CLASS  ********************

        Parameters
        ----------
        search_depth : int (optional)
            A strictly positive integer (i.e., 1, 2, 3,...) for the number of
            layers in the game tree to explore for fixed-depth search. (i.e., a
            depth of one (1) would only explore the immediate sucessors of the
            current state.)

        score_fn : callable (optional)
            A function to use for heuristic evaluation of game states.

        timeout : float (optional)
            Time remaining (in milliseconds) when search is aborted. Should be a
            positive value large enough to allow the function to return before the
            timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
        search. You must finish and test this player to make sure it properly uses
        minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
            result before the time limit expires.

            **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

            For fixed-depth search, this function simply wraps the call to the
            minimax method, but this method provides a common interface for all
            Isolation agents, and you will replace it in the AlphaBetaPlayer with
            iterative deepening search.

            Parameters
            ----------
            game : `isolation.Board`
                An instance of `isolation.Board` encoding the current state of the
                game (e.g., player locations and blocked cells).

            time_left : callable
                A function that returns the number of milliseconds left in the
                current turn. Returning with any less than 0 ms remaining forfeits
                the game.

            Returns
            -------
            (int, int)
                Board coordinates corresponding to a legal move; may return
                (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        best_move = (-1, -1)

        # Try to explore as deep as possible
        try:
            return self.minimax(game, self.search_depth)
        except SearchTimeout:
            pass

        return best_move

    def terminal_test(self, game):
        """ Return True if the game is over for the active player and False otherwise."""
        if self.time_left() < self.TIMER_THRESHOLD:
             raise SearchTimeout()

        return len(game.get_legal_moves()) == 0

    def max_value(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if self.terminal_test(game):
            return -1

        if depth <= 0:
            return self.score(game, self)

        v = float("-inf")
        for move in game.get_legal_moves():
            v = max(v, self.min_value(game.forecast_move(move), depth - 1))
        return v

    def min_value(self, game, depth):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if self.terminal_test(game):
            return 1

        if depth <= 0:
            return self.score(game, self)

        v = float("inf")
        for move in game.get_legal_moves():
            v = min(v, self.max_value(game.forecast_move(move), depth - 1))
        return v

    def minimax(self, game, depth):
        """
            Parameters
            ----------
            game : isolation.Board
                An instance of the Isolation game `Board` class representing the
                current game state

            depth : int
                Depth is an integer representing the maximum number of plies to
                search in the game tree before aborting

            Returns
            -------
            float
                The score

            (int, int)
                The board coordinates of the best move found in the current search;
                (-1, -1) if there are no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_score, best_move = float("-inf"), (-1, -1)

        # Early exit when no legal moves
        if self.terminal_test(game):
            return best_move

        for move in game.get_legal_moves():
            v = self.min_value(game.forecast_move(move), depth - 1)
            if v > best_score:
                best_score, best_move = v, move

        return best_move  
                

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
        search with alpha-beta pruning. You must finish and test this player to
        make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
            result before the time limit expires.

            Modify the get_move() method from the MinimaxPlayer class to implement
            iterative deepening search instead of fixed-depth search.

            **********************************************************************
            NOTE: If time_left() < 0 when this function returns, the agent will
                  forfeit the game due to timeout. You must return _before_ the
                  timer reaches 0.
            **********************************************************************

            Parameters
            ----------
            game : `isolation.Board`
                An instance of `isolation.Board` encoding the current state of the
                game (e.g., player locations and blocked cells).

            time_left : callable
                A function that returns the number of milliseconds left in the
                current turn. Returning with any less than 0 ms remaining forfeits
                the game.

            Returns
            -------
            (int, int)
                Board coordinates corresponding to a legal move; may return
                (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        best_move = (-1, -1)

        # Try to explore as deep as possible
        try:
            depth = 1
            while True:
                best_move = self.alphabeta(game, depth)
                depth += 1
                
        except SearchTimeout:
            pass

        return best_move

    def terminal_test(self, game):
        """ Return True if the game is over for the active player and False otherwise."""
        if self.time_left() < self.TIMER_THRESHOLD:
             raise SearchTimeout()

        return len(game.get_legal_moves()) == 0

    def max_value(self, game, depth, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if self.terminal_test(game):
            return -1

        if depth <= 0:
            return self.score(game, self)

        v = float("-inf")
        for move in game.get_legal_moves():
            v = max(v, self.min_value(game.forecast_move(move), depth - 1, alpha, beta))

            # If possible escape the "for" loop
            if v >= beta:
                return v
            alpha = max(alpha, v) 

        return v

    def min_value(self, game, depth, alpha, beta):
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        if self.terminal_test(game):
            return 1

        if depth <= 0:
            return self.score(game, self)

        v = float("inf")
        for move in game.get_legal_moves():
            v = min(v, self.max_value(game.forecast_move(move), depth - 1, alpha, beta))

            # If possible escape the "for" loop
            if v <= alpha:
                return v
            beta = min(beta, v)

        return v

    def alphabeta(self, game, depth):
        """Implement depth-limited minimax search with alpha-beta pruning as
            described in the lectures.

            Parameters
            ----------
            game : isolation.Board
                An instance of the Isolation game `Board` class representing the
                current game state

            depth : int
                Depth is an integer representing the maximum number of plies to
                search in the game tree before aborting

            alpha : float
                Alpha limits the lower bound of search on minimizing layers

            beta : float
                Beta limits the upper bound of search on maximizing layers

            Returns
            -------
            (int, int)
                The board coordinates of the best move found in the current search;
                (-1, -1) if there are no legal moves

        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        best_score, best_move = float("-inf"), (-1, -1)
        alpha, beta = float("-inf"), float("inf")

        if depth <= 0:
            return best_move

        for move in game.get_legal_moves():
            v = self.min_value(game.forecast_move(move), depth - 1, alpha, beta)
            if v > best_score:
                best_score, best_move = v, move

            alpha = max(alpha, v) 

        return best_move