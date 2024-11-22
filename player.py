from state import State
import random

class Player:
    def __init__(self, name):
        self.name = name
    
    def make_move(self, state:State):
        row = random.choice([0,1,2])
        column = random.choice([0,1,2])
        return (row, column)