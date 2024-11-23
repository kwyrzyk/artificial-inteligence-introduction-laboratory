from state import State
from player import Player
from game import Game

player1 = Player("Player1", 10, True)
player2 = Player("Player2", 1, False)
game = Game(player1, player2)
game.start()