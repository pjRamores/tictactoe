
class BotPlayer:
    def check_winner(self, board, player):
        # Check rows
        for row in range(3):
            if all(board[row*3+cell] == player for cell in range(3)):
                return True
        # for row in board:
        #     if all(cell == player for cell in row):
        #         return True
        # Check columns
        for col in range(3):
            if all(board[row*3+col] == player for row in range(3)):
                return True
        # Check diagonals
        if all(board[i*3+i] == player for i in range(3)):
            return True
        if all(board[i*3+(2-i)] == player for i in range(3)):
            return True
        return False

    def is_board_full(self, board):
        return all(board[row*3+cell] != 0 for row in range(3) for cell in range(3))

    def get_empty_cells(self, board):
        return [(i, j) for i in range(3) for j in range(3) if board[i*3+j] == 0]

    def minimax(self, board, depth, is_maximizing, bot_player, opponent):
        if self.check_winner(board, bot_player):
            return 1
        if self.check_winner(board, opponent):
            return -1
        if self.is_board_full(board):
            return 0

        if is_maximizing:
            best_score = -float('inf')
            for i, j in self.get_empty_cells(board):
                board[i*3+j] = bot_player
                score = self.minimax(board, depth + 1, False, bot_player, opponent)
                board[i*3+j] = 0
                best_score = max(score, best_score)
            return best_score
        else:
            best_score = float('inf')
            for i, j in self.get_empty_cells(board):
                board[i*3+j] = opponent
                score = self.minimax(board, depth + 1, True, bot_player, opponent)
                board[i*3+j] = 0
                best_score = min(score, best_score)
            return best_score

    def get_best_move(self, board, bot_player, opponent):
        best_score = -float('inf')
        best_move = None
        for i, j in self.get_empty_cells(board):
            board[i*3+j] = bot_player
            score = self.minimax(board, 0, False, bot_player, opponent)
            board[i*3+j] = 0
            if score > best_score:
                best_score = score
                best_move = (i, j)
        return best_move
