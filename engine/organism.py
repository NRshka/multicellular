from torch import FloatTensor as Tensor

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
            vectors.append([i.item() for i in cell.action(vision)])

        averaged_direction: Tensor = Tensor(vectors).mean(0)

        for cell in self.muscles:
            cell.x += averaged_direction[0]
            cell.y += averaged_direction[1]
            #bacward pass
            cell.brain.backward(averaged_direction)
