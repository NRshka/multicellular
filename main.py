from torch import zeros
from math import sin, cos, pi

from engine import Muscle, Organism




if __name__ == "__main__":
    moving_cells = [Muscle(cos(2*pi*i/4), sin(2*pi*i/4)) for i in range(4)]

    for cell in moving_cells:
        cell.add_links(moving_cells)

    org = Organism(moving_cells)
    vision = zeros((1, 1, 128, 128))
    org.move(vision)
