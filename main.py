from state import State
from player import Player
from player import GameStyle
from display import Display
from game import Game

player1 = Player("Player1", 0, GameStyle.MAXIMIZING)
player2 = Player("Player2", 1, GameStyle.MINIMIZING)
display = Display()
game = Game(player1, player2, display)
game.start()