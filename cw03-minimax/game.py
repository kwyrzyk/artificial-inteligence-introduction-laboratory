from state import State
from player import Player
from display import Display


class Game:
    def __init__(self, player1: Player, player2: Player, display: Display = None):
        self._player1 = player1
        self._player2 = player2
        self._next_player = player1
        self._state = State()
        self._display = display

    def start(self) -> None:
        while not self._state.is_finished():
            player_move = self._next_player.make_move(self._state)
            self._state.mark_cell(*player_move)
            self.switch_player()
            if self._display is not None:
                self._display.show_state(self._state, self._next_player)
        if self._display is not None:
            self._display.show_result(self._state, self._player1, self._player2)

    def switch_player(self) -> None:
        if self._next_player == self._player1:
            self._next_player = self._player2
        else:
            self._next_player = self._player1

    def new_game(self) -> None:
        self._next_player = self._player1
        self._state = State()

    def get_player1(self) -> Player:
        return self._player1

    def set_player1(self, new_player: Player) -> None:
        self._player1 = new_player

    def get_player2(self) -> Player:
        return self._player2

    def set_player2(self, new_player: Player) -> None:
        self._player2 = new_player

    def get_next_player(self) -> Player:
        return self._next_player

    def set_next_player(self, new_player: Player) -> None:
        self._next_player = new_player

    def get_state(self) -> State:
        return self._state

    def set_state(self, new_state: State) -> None:
        self._state = new_state

    def get_display(self) -> Display:
        return self._display

    def set_display(self, new_display: Display) -> None:
        self._display = new_display
