from typing import Union
from dataclasses import dataclass
from overrides import overrides


from .intelligence.nn import MovingBrain


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



class Muscle(Cell):
    '''
    Moving cell
    '''
    def __init__(self, x, y, links: list = []):
        super().__init__(hp=10., energy=100., links=links, x=x, y=y)
        self.brain = MovingBrain()


    def action(self, vision) -> list:
        '''
        Do moving according AI
        '''
        return list(self.brain.action(vision))
