from torch import FloatTensor as Tensor
from math import sqrt

from .cells import Muscle


def norm(vec: tuple) -> float:
    return sqrt(sum([v**2 for v in vec]))


def scalar_mul(a: tuple, b: tuple) -> float:
    '''
    Scalar multiplication function of two vectors

    @param a: one vector
    @param b: another vectors
    @return scalar multiplication
    '''
    assert len(a) == len(b), ValueError("Dimensions of vectors must be same")

    return sum([ai*bi for ai, bi in zip(a, b)])


class Organism:
    '''
    '''
    def __init__(self, muscles):
        self.muscles = muscles
        self.speed = 500.0


    def move(self, vision):
        vectors: list = []
        for cell in self.muscles:
            vectors.append([i.item() for i in cell.action(vision)])

        averaged_direction: Tensor = Tensor(vectors).mean(0).detach()
        self.vectors = vectors

        for ind, cell in enumerate(self.muscles):
            #for neibor in cell.links:
            #    cell.x += vector[0].
            if cell.energy <= 0:
                del self.muscles[ind]
                continue

            dx: float = self.vectors[ind][0]
            dy: float = self.vectors[ind][1]

            for neibor in cell.links:
                link_vec: tuple = (cell.x - neibor.x, cell.y - neibor.y)
                norm_l: float = norm(link_vec)
                
                if norm_l == 0:
                    continue
                
                proj_len: float = scalar_mul(link_vec, (dx, dy))
                projection: tuple = (link_vec[0]*proj_len/norm_l, link_vec[1]*proj_len/norm_l)

                neibor.x += projection[0]
                neibor.y += projection[1]

            cell.x += self.speed * dx
            cell.y += self.speed * dy
            #bacward pass
            cell.brain.backward(averaged_direction)

    def get_average_coord(self):
        average_x = 0.
        average_y = 0.

        for cell in self.muscles:
            average_x += cell.x
            average_y += cell.y

        return average_x / len(self.muscles), average_y / len(self.muscles)
