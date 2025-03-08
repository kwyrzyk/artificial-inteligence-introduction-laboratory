from copy import deepcopy


class State:
    def __init__(self) -> None:
        self._next_move = 1
        self._moves_made = 0
        self._board = [[0 for _ in range(3)] for _ in range(3)]

    def get_next_move(self) -> int:
        return self._next_move

    def get_moves_made(self) -> int:
        return self._moves_made

    def get_board(self) -> list[list[int]]:
        return self._board

    def switch_player(self) -> None:
        self._next_move = -self._next_move

    def mark_cell(self, row: int, column: int) -> None:
        self._board[row][column] = self._next_move
        self._moves_made += 1
        self.switch_player()

    def row_sum(self, row_num: int) -> int:
        return sum(self._board[row_num])

    def column_sum(self, column_num: int) -> int:
        return sum([self._board[row_num][column_num] for row_num in range(3)])

    def is_won(self) -> bool:
        for row_num in range(3):
            row_sum = self.row_sum(row_num)
            if row_sum in [-3, 3]:
                return True

        for column_num in range(3):
            column_sum = self.column_sum(column_num)
            if column_sum in [-3, 3]:
                return True

        diagonal1_sum = self._board[0][0] + self._board[1][1] + self._board[2][2]
        diagonal2_sum = self._board[0][2] + self._board[1][1] + self._board[2][0]
        if diagonal1_sum in [-3, 3] or diagonal2_sum in [-3, 3]:
            return True
        return False

    def is_finished(self) -> bool:
        if self._moves_made == 9:
            return True
        return self.is_won()

    def get_result(self) -> Optional[int]:
        if not self.is_finished():
            return None
        else:
            if self.is_won():
                return -self._next_move
            else:
                return 0

    def get_successors_with_moves(self) -> dict["State", tuple[int, int]]:
        successors = {}
        for row in range(3):
            for column in range(3):
                if self._board[row][column] == 0:
                    successor = deepcopy(self)
                    successor.mark_cell(row, column)
                    move = (row, column)
                    successors[successor] = move
        return successors

    def get_successors(self) -> list["State"]:
        return list(self.get_successors_with_moves().keys())

    def heuristic(self) -> int:
        if self.is_won():
            if self._next_move == -1:
                return 100
            else:
                return -100
        corners_sum = (
            self._board[0][0]
            + self._board[0][2]
            + self._board[2][0]
            + self._board[2][2]
        )
        edges_sum = (
            self._board[0][1]
            + self._board[1][0]
            + self._board[1][2]
            + self._board[2][1]
        )
        center_sum = self._board[1][1]
        heuristic_value = 4 * center_sum + 3 * corners_sum + 2 * edges_sum
        return heuristic_value

    def row_repr(self, values: list[int]) -> str:
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

    def __repr__(self) -> str:
        rows = []
        separating_line = "-" * 11 + "\n"
        for row in self._board:
            rows.append(self.row_repr(row))
        repr = separating_line.join(rows)
        repr = repr[:-2]
        return repr
