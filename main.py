from state import State
from player import Player
from display import Display
from game import Game

player1 = Player("Player1", 8)
player2 = Player("Player2", 2)
display = Display()
game = Game(player1, player2, display)
game.start()

# Usunąc gamestyl