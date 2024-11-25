from state import State
from player import Player
from display import Display
from game import Game

GAMES_NUMBER = 1000
MAX_X_DEPTH = 2
MAX_O_DEPTH = 2

for x in range(1, MAX_X_DEPTH + 1):
    for o in range(1, MAX_O_DEPTH + 1):
        player1 = Player("Player1", x)
        player2 = Player("Player2", o)
        game = Game(player1, player2)
        x_wins = 0
        draws = 0
        o_wins = 0
        for _ in range(GAMES_NUMBER):
            game.start()
            result = game.get_state().get_result()
            match result:
                case 1:
                    x_wins += 1
                case 0:
                    draws += 1
                case -1:
                    o_wins += 1
            game.new_game()
        print("X depth =", x)
        print("O depth =", o)
        print(player1.get_name(), "won", x_wins, "times")
        print("Game ended in a draw", draws, "times")
        print(player2.get_name(), "won", o_wins, "times")
        print("-" * 20)



