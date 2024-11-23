from state import State
from player import Player


class Game:
    def __init__(self, player1:Player, player2:Player):
        self.player1 = player1
        self.player2 = player2
        self.next_player = player1
        self.state = State()
    
    def start(self):
        while not self.state.is_finished():
            player_move = self.next_player.make_move(self.state)
            self.state.mark_cell(*player_move)
            self.switch_player()
            print(self.state)
            print(self.state.heuristic())

    def switch_player(self):
        if self.next_player == self.player1:
            self.next_player = self.player2
        else:
            self.next_player = self.player1
        
