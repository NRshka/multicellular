import numpy as np

from .cells import Muscle



class Organism:
    '''
    '''
    def __init__(self, muscles):
        self.muscles = muscles
        self.speed = 1.0


    def move(self, vision):
        vectors: list = []
        for cell in self.muscles:
            vectors.append(cell.action(vision))

        averaged_direction: np.array = np.array(vectors).mean(0)

        for cell in self.muscles:
            cell.x += averaged_direction[0]
            cell.y += averaged_direction[1]
