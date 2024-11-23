from copy import deepcopy
import sys


class State:
    def __init__(self):
        self.next_move = 1
        self.moves_made = 0
        self.board = [[0 for _ in range(3)] for _ in range (3)]
    
    def get_next_move(self):
        return self.next_move

    def switch_player(self):
        self.next_move = -self.next_move
    
    def get_board(self):
        return self.board
    
    def mark_cell(self, row:int, column:int):
        self.board[row][column] = self.next_move
        self.moves_made += 1
        self.switch_player()

    def row_sum(self, row_num:int):
        return sum(self.board[row_num])
    
    def column_sum(self, column_num:int):
        return sum([self.board[row_num][column_num] for row_num in range(3)])

    def is_won(self):
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
    
    def is_finished(self):
        if self.moves_made == 9:
            return True
        return self.is_won()

    def get_successors_with_moves(self):
        succesors = {}
        for row in range(3):
            for column in range(3):
                if self.board[row][column] == 0:
                    succesor = deepcopy(self)
                    succesor.mark_cell(row, column)
                    move = (row, column)
                    succesors[succesor] = move
        return succesors
    
    def get_successors(self):
        return list(self.get_successors_with_moves().keys())
    
    def heuristic(self):
        if self.is_won():
            if self.next_move == -1:
                return 100
            else:
                return -100
        corners_sum = self.board[0][0] + self.board[0][2] + self.board[2][0] + self.board[2][2]
        edges_sum = self.board[0][1] + self.board[1][0] + self.board[1][2] + self.board[2][1]
        center_sum = self.board[1][1]
        heuristic_value = 4 * center_sum + 3 * corners_sum + 2 * edges_sum
        return heuristic_value

    def row_repr(self, values:list):
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