from typing import Union
from dataclasses import dataclass
from overrides import overrides


from .intelligence.nn import MovingBrain, RangeAttackBrain


@dataclass
class Cell:
    hp: float
    energy: float
    links: list
    x: float
    y: float

    def add_links(self, cells: Union[list, set]) -> None:
        '''
        Adds links to the cells to which the cell is linked.
        References can't be repeated - it's set.
        Cell can't be linked with itself.
        
        @param cells: list or set of linked cells
        @return None
        '''
        for cell in cells:
            if id(cell) != id(self):
                self.links.append(cell)


    def got_bitten(self, dmg):
        self.hp -= dmg


class Muscle(Cell):
    '''
    Moving cell
    '''
    def __init__(self, x, y, links: list = []):
        super().__init__(hp=10., energy=100., links=links, x=x, y=y)
        self.brain = MovingBrain()


    def action(self, vision) -> list:
        '''
        Does moving according AI
        '''
        direction = self.brain.action(vision)
        self.energy -= 1.*direction.mean()

        return direction


class Jaw(Cell):
    def __init__(self, x, y, range, links: list = []):
        super().__init__(hp=50., energy=100., links=links, x=x, y=y)
        self.brain = RangeAttackBrain()
        self.range = range


    def action(self, vision) -> list:
        '''
        Does attack according AI
        '''
        direction = self.brain.action(vision)
        if direction:
            self.energy -= 2*direction.mean()

        return self.range * direction