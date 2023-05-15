class TicTacToe:
    def __init__(self):
        self.board = [' '] * 9  # Initialize an empty board
        self.current_player = 'X'  # Player 'X' starts the game

    def print_board(self):
        print("-------------")
        for i in range(0, 9, 3):
            print(f"| {self.board[i]} | {self.board[i+1]} | {self.board[i+2]} |")
            print("-------------")

    def make_move(self, position):
        if self.board[position] == ' ':
            self.board[position] = self.current_player
            self.current_player = 'O' if self.current_player == 'X' else 'X'
        else:
            print("Invalid move. Try again.")

    def check_winner(self):
        winning_positions = [
            [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Rows
            [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Columns
            [0, 4, 8], [2, 4, 6]  # Diagonals
        ]

        for position in winning_positions:
            if self.board[position[0]] == self.board[position[1]] == self.board[position[2]] != ' ':
                return self.board[position[0]]  # Return the winning player

        if ' ' not in self.board:
            return 'Draw'  # If the board is full and no winner, it's a draw

        return None  # If no winner yet

    def play_game(self):
        while True:
            self.print_board()
            position = int(input("Enter the position to make a move (0-8): "))
            self.make_move(position)
            winner = self.check_winner()

            if winner:
                if winner == 'Draw':
                    print("It's a draw!")
                else:
                    print(f"Player {winner} wins!")
                break

game = TicTacToe()
game.play_game()
