import numpy as np

class TicTacToe:
    def __init__(self):
        self.board = np.zeros(9, dtype=int)  # 3x3 board flattened
        # self.board = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=int)
        self.done = False
        self.winner = None

    def reset(self):
        self.board = np.zeros(9, dtype=int)
        # self.board = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8], dtype=int)
        self.done = False
        self.winner = None
        return self.board.copy()

    def get_valid_moves(self):
        return [i for i in range(9) if self.board[i] == 0]
        # return [i for i in range(9) if self.board[i] != 10 and self.board[i] != 11]

    def step(self, action, player=1):
        if action not in self.get_valid_moves():
            return self.board.copy(), -1, True, "Invalid move"
        self.board[action] = player
        reward, done = self.check_game_over()
        return self.board.copy(), reward, done, ""

    def check_game_over(self):
        win_combinations = [
            [0,1,2], [3,4,5], [6,7,8],  # Rows
            [0,3,6], [1,4,7], [2,5,8],  # Columns
            [0,4,8], [2,4,6]            # Diagonals
        ]
        for combo in win_combinations:
            if self.board[combo[0]] == self.board[combo[1]] == self.board[combo[2]] != 0:
                self.done = True
                self.winner = self.board[combo[0]]
                return self.winner, True
                # return 1 if self.winner == 10 else -1, True
        if 0 not in self.board:
        # if all(i > 8 for i in self.board):
            self.done = True
            self.winner = 0
            return 0, True
        return 0, False

    def render(self):
        # Display the board in a 3x3 grid
        # symbols = {1: 'X', -1: 'O', 0: ' '}
        symbols = {1: 'X', -1: 'Y', 0: ' '}
        # symbols = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 10: 'M', 11: 'H'}
        index = 0
        for i in range(3):
            print('|', end='')
            for j in range(3):
                symbol = symbols[self.board[i*3+j]]
                if symbol == ' ':
                    symbol = str(index)
                print(f' {symbol} |', end='')
                index += 1
            print('\n' + '-'*11)
        print()