from typing import Type
from collections import namedtuple
import os.path
import sys
import sympy
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

GameState: Type[GameState] = namedtuple('GameState', 'to_move, utility, board, moves')

n = 7
stones = []
taken = []
depth = 0
lastInput = 6
mymoves = []
myvalue=0


def processInput(myInput):
    global n
    global stones
    global taken
    global depth
    global lastInput
    myTemp = []

    number_string = myInput.split(' ')
    intInput = [int(i) for i in number_string]
    n = intInput[0]
    for i in range(n):
        stones.append(i+1)
        taken.append(0)
    if intInput[1] != 0:
        for i in range(intInput[1]):
            myTemp.append(0)
            myTemp[i] = intInput[i+2]
        for i in range(len(myTemp)):
            taken[myTemp[i] - 1] = 1
        lastInput = myTemp[-1]
    depth = intInput[-1]
    if len(myTemp) % 2 != 0:
        return "MIN"
    else:
        return "MAX"

# Used from AIMA Code
class Game:
    """A game is similar to a problem, but it has a utility for each
    state and a terminal test instead of a path cost and a goal
    test. To create a game, subclass this class and implement actions,
    result, utility, and terminal_test. You may override display and
    successors or you can inherit their default methods. You will also
    need to set the .initial attribute to the initial state; this can
    be done in the constructor."""

    def actions(self, state):
        """Return a list of the allowable moves at this point."""
        raise NotImplementedError

    def result(self, state, move):
        """Return the state that results from making a move from a state."""
        raise NotImplementedError

    def utility(self, state, player):
        """Return the value of this final state to player."""
        raise NotImplementedError

    def terminal_test(self, state):
        """Return True if this is a final state for the game."""
        return not self.actions(state)

    def to_move(self, state):
        """Return the player whose move it is in this state."""
        return state.to_move

    def display(self, state):
        """Print or otherwise display the state."""
        print(state)

    def __repr__(self):
        return '<{}>'.format(self.__class__.__name__)

    def play_game(self, *players):
        """Play an n-person, move-alternating game."""
        state = self.initial
        while True:
            for player in players:
                move = player(self, state)
                state = self.result(state, move)
                if self.terminal_test(state):
                    self.display(state)
                    return self.utility(state, self.to_move(self.initial))

#Used in AIMA Code, restructured and adjusted
class TakeStones(Game):
    def __init__(self):
        moves = []
        board = []
        for i in range(n):
            moves.append(i + 1)
            board.append(0)
        self.initial = GameState(to_move="MAX", utility=0, board=board, moves=moves)

    def actions(self, state):
        board = state.board
        moves = []
        if sum(board) == 0:  # First turn restrictions
            for i in range(n):
                if (state.moves[i] <= np.floor(n / 2)) and (state.moves[i] % 2 != 0):
                    moves.append(state.moves[i])
            return moves
        else:
            for i in state.moves:  # Subsequent turn restrictions
                if board[i - 1] == 0:
                    if ((lastInput % state.moves[i - 1]) == 0) or ((state.moves[i - 1] % lastInput) == 0):
                        moves.append(state.moves[i - 1])
        return moves

    def result(self, state, move):
        global mymoves
        global lastInput
        if move not in state.moves:
            return state  # Illegal move has no effect
        board = state.board.copy()
        # board[move] = state.to_move
        moves = list(state.moves)
        moves.remove(move)
        mymoves = moves
        return GameState(to_move=('MIN' if state.to_move == 'MAX' else 'MAX'),
                         utility=self.compute_utility(board, move, state.to_move),
                         board=board, moves=moves)

    def utility(self, state, player):
        return state.utility if player == 'MAX' else -state.utility

    def terminal_test(self, state):
        return state.utility != 0 or len(state.moves) == 0

    def display(self, state):
        board = state.board
        print(f"{board}\n")

    def compute_utility(self, board, move, player):
        global myvalue
        value = 0
        if board[0] == 0:
            value = 0.1
        elif lastInput == 1:
            if len(mymoves) % 2 == 0:
                value = 0.5
            else:
                value = -0.5
        elif sympy.isprime(lastInput):
            if len(mymoves) % 2 == 0:
                value = 0.7
            else:
                value = -0.7
        else:
            temp = sympy.nextprime(lastInput)
            count = 0
            if temp <= n:
                for i in mymoves:
                    if (mymoves[i] % temp == 0) or temp % mymoves[i] == 0:
                        count = count + 1
                if count % 2 == 0:
                    value = 0.6
                else:
                    value = -0.6
        myvalue=value
        if player == 'MAX':
            return value
        else:
            return -value


# used from AIMA Code
def alpha_beta_search(state, game):
    """Search game to determine best action; use alpha-beta pruning.
    As in [Figure 5.7], this version searches all the way to the leaves."""

    player = game.to_move(state)

    # Functions used by alpha_beta
    def max_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = -np.inf
        for a in game.actions(state):
            v = max(v, min_value(game.result(state, a), alpha, beta))
            if v >= beta:
                return v
            alpha = max(alpha, v)
        return v

    def min_value(state, alpha, beta):
        if game.terminal_test(state):
            return game.utility(state, player)
        v = np.inf
        for a in game.actions(state):
            v = min(v, max_value(game.result(state, a), alpha, beta))
            if v <= alpha:
                return v
            beta = min(beta, v)
        return v

    # Body of alpha_beta_search:
    best_score = -np.inf
    beta = np.inf
    best_action = None
    for a in game.actions(state):
        v = min_value(game.result(state, a), best_score, beta)
        if v > best_score:
            best_score = v
            best_action = a
    return best_action


def alpha_beta_player(game, state):
    return alpha_beta_search(state, game)


def main():
    temp = input("Please enter input:\n")
    processInput(temp)
    TS = TakeStones()
    # TS.display(TS.initial)
    my_state = GameState(to_move='MAX', utility='0', board=taken, moves=stones)
    # TS.display(my_state)
    mymove = alpha_beta_player(TS, my_state)
    print(f"Move:{mymove}\nValue:{myvalue}\nNumber of Nodes Visited:{depth}")

    return 0

"""Inputs:
    7 3 1 4 2 3
    7 2 3 6 0
    7 3 2 1 3 6 
    7 2 3 6 1
"""


if __name__ == "__main__":
    main()
