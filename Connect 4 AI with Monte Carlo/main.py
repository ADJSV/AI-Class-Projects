from MCTS import UCT_ver1
from copy import deepcopy

#From public repo of Somewheref
# hyperparameters
WINNING_REWARD = 1
TIE_REWARD = 0
LOSING_REWARD = 0
VERBOSE = False

horizontal = 7
vertical = 6

# directions to search when determine whether player wins
serach_directions = [[[1, 0]],  # down
                     [[0, -1], [0, 1]],  # left and right
                     [[1, -1], [-1, 1]],  # left-down to right-up
                     [[-1, -1], [1, 1]]]  # left-up to right-down
totalMoves=[]
class Board:
    def __init__(self, to_play):
        self.board=[[0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]]  # Create a 2D array of 6x7 of 0s
        self.current_player=to_play

    #From public repo of Somewheref
    def get_possible_actions(self, state):
        # unpack state if it is a string
        if type(state) == str:
            state = self.unpack_state(state)
        possible_actions = []
        for i, cell in enumerate(state[0]):
            if not cell:
                possible_actions.append(i + 1)
        return possible_actions

    def next_state(self, action, board=None, player=None):
        action -= 1
        if board is None:
            board = deepcopy(self.board)
        elif type(board) == str:
            board = self.unpack_state(board)

        if player is None:
            player = self.current_player

        for line in range(vertical - 1, -1, -1):
            # from bottom to top
            if board[line][action]:  # continue if this grid is occupied
                continue
            else:  # finds an empty grid
                # copy the board
                board[line][action] = player
                return self.get_compact_state(board)

    def get_compact_state(self, board):
        'convert a list to compact string'
        string = ''
        for line in board:
            for char in line:
                string += str(char)
            string += '|'
        return string[:-1]

    def unpack_state(self, board):
        new_board = board.split('|')
        for i, line in enumerate(new_board):
            new_board[i] = [0 if x == "0" else x for x in line]
        return new_board

    def check_winning(self, pos, board=None, player=None):
        """check if game ends. return 0 if draw, +reward if wins and -reward if loses
        if row is a tuple (exact coordinate), than will use that for coordinate. Otherwise automatically detect for pos by considering it as row.
        if param board is not passed, then will use current board in this instance
        if param player is not passed, then will use current player"""

        if board is None:
            board = self.board
        elif type(board) == str:
            board = self.unpack_state(board)

        if player is None:
            player = "X" if self.current_player == "O" else "O"

        # find position if not specified
        if type(pos) != tuple:
            pos -= 1
            row = [line[pos] for line in board]
            line = [i + 1 for i, x in enumerate(row[1:]) if not row[i] and x]
            if len(line) == 0:
                pos = (0, pos)
            else:
                pos = (line[0], pos)
            last_player = board[pos[0]][pos[1]]
        else:
            last_player = board[pos[0]][pos[1]]

        for direction in serach_directions:
            total = 0  # counter recording how many checkers found in a line
            for dirt in direction:
                try:
                    for i in range(1, 4):
                        # in case for negative index value (which is a special feature in python but detrimental here)
                        if pos[0] + dirt[0] * i < 0 or pos[1] + dirt[1] * i < 0:
                            break
                        if board[pos[0] + dirt[0] * i][
                            pos[1] + dirt[1] * i] == last_player:  # checker of current player found
                            total += 1
                        else:
                            break
                    if total >= 3:
                        if player == last_player:
                            return WINNING_REWARD
                        return LOSING_REWARD
                except IndexError:
                    continue

        # check for draw
        if all(board[0]):
            return TIE_REWARD
        return False

    def next_player(self, player=None):
        if player is None:
            player = self.current_player
        return "X" if player == "O" else "O"

def processInput():
    global totalMoves
    data=list(input("Enter state: \n").replace(" ",""))
    if data[0]=='e':
        return 1
    for i in range(len(data)):
        if (data[i]!='X') & (data[i]!='O') & (data[i]!='0'):
            print("invalid input")
            return -1
    myinput=[]
    counter=0
    for i in range(6):
        myinput.append([])
        for j in range(7):
            myinput[i].append(data[counter])
            counter=counter+1
    #Checking for winners already
    print(myinput)
    for i in range(6):
        for j in range(7):
            if i >0:
                if (myinput[5 - i][6 - j] != "0") & (myinput[5 + 1 - i][6 - j] == "0"):
                    print(f"Not allowed, floating move at position {5 - i},{6 - j}")
                    return -1
            if (5 - i) < 3:
                if (6 - j) < 4:
                    if (myinput[5 - i][6 - j] == myinput[5 + 1 - i][6 + 1 - j] == myinput[5 + 2 - i][6 + 2 - j] == \
                        myinput[5 + 3 - i][6 + 3 - j]) & (myinput[5 - i][6 - j] != "0"):
                        print(f"Already a winner {myinput[5 - i][6 - j]}")  # diagDown
                        return -1
            if (5 - i) >= 3:
                if (6 - j) < 4:
                    if (myinput[5 - i][6 - j] == myinput[5 - 1 - i][6 + 1 - j] == myinput[5 - 2 - i][6 + 2 - j] == \
                        myinput[5 - 3 - i][6 + 3 - j]) & (myinput[5 - i][6 - j] != "0"):
                        print(f"Already a winner {myinput[5 - i][6 - j]}")  # diagUp
                        return -1
                if (myinput[5 - i][6 - j] == myinput[5 - 1 - i][6 - j] == myinput[5 - 2 - i][6 - j] == \
                    myinput[5 - 3 - i][6 - j]) & (myinput[5 - i][6 - j] != "0"):
                    print(f"Already a winner {myinput[5 - i][6 - j]}")  # vertical
                    return -1
            if (6 - j) <= 3:
                if (myinput[5 - i][6 - j] == myinput[5 - i][6 + 1 - j]) & (
                        myinput[5 - i][6 - j] == myinput[5 - i][6 + 2 - j]) & (
                        myinput[5 - i][6 - j] == myinput[5 - i][6 + 3 - j]) & (myinput[5 - i][6 - j] != "0"):
                    print(f"Already a winner {myinput[5 - i][6 - j]}")  # horizontal
                    return -1

    board = Board(to_play='X')
    board.board=myinput
    ai_1 = UCT_ver1(board, name='ai1', verbose=VERBOSE)
    ai_1.update(board.get_compact_state(board.board), "X")
    action = ai_1.get_action()
    totalMoves.append(action)
    print(action)
    del ai_1

def convertData():
    myinput=[]
    for i in range(6):
        myinput.append([])
        for j in range(7):
            myinput[i].append(0)

    data=list(input(">"))
    if data[0]=='e':
        return 1
    for i in range(len(data)):
        data[i]=int(data[i])

    ch="O"
    for i in range(len(data)):
        if ch=="X": ch="O"
        else: ch="X"
        for j in range(6):
            if myinput[j][data[i]-1]==0:
                myinput[j][data[i]-1]=ch
                break
    for i in range(6):
        for j in range(7):
            print(f"{myinput[5-i][6-j]}",end="")
    print("\n")
if __name__ == "__main__":

    toQuit=0
    while not toQuit:
        #toQuit=convertData()
        toQuit=processInput()
    print(totalMoves)

