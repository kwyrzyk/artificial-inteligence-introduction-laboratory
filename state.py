class State:
    def __init__(self):
        self.next_move = 1
        self.board = [[0 for _ in range(3)] for _ in range (3)]
    
    def get_next_move(self):
        return self.next_move

    def switch_player(self):
        self.next_move = -self.next_move
    
    def get_board(self):
        return self.board
    
    def mark_cell(self, row, column):
        self.board[row][column] = self.next_move
        self.switch_player()

    def row_sum(self, row_num):
        return sum(self.board[row_num])
    
    def column_sum(self, column_num):
        return sum([self.board[row_num][column_num] for row_num in range(3)])

    def is_finished(self):
        for row_num in range(3):
            row_sum = self.row_sum(row_num)
            if row_sum in [-3,3]:
                return True
            
        for column_num in range(3):
            column_sum = self.column_sum(column_num)
            if column_sum in [-3,3]:
                return True
            
        diagonal1_sum = self.board[0][0] + self.board[1][1] + self.board[2][2]
        diagonal2_sum = self.board[0][2] + self.board[1][1] + self.board[2][0]
        if diagonal1_sum in [-3, 3] or diagonal2_sum in [-3,3]:
            return True
        
        return False

    def row_repr(self, values):
        symbols = []
        for value in values:
            match value:
                case 1:
                    symbols.append(" X ")
                case 0:
                    symbols.append("   ")
                case -1:
                    symbols.append(" O ")
        repr = "|".join(symbols) + "\n"
        return repr
    
    def __repr__(self):
        rows = []
        separating_line = "-" * 11 + "\n"
        for row in self.board:
            rows.append(self.row_repr(row))
        repr = separating_line.join(rows)
        return repr