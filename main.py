from torch import zeros
from math import sin, cos, pi, sqrt
import arcade

from engine import Muscle, Organism


SCREEN_WIDTH = 600
SCREEN_HEIGHT = 600

# Открыть окно. Задать заголовок и размеры окна (ширина и высота)
arcade.open_window(SCREEN_WIDTH, SCREEN_HEIGHT, "Drawing Example")


def on_draw(delta_time):
    vision = zeros((1, 1, 128, 128))
    arcade.start_render()
    for org, color in zip(on_draw.organisms, on_draw.colors):
        org.move(vision)

        for cell, vec in zip(org.muscles, org.vectors):
            arcade.draw_circle_filled(cell.x, cell.y, 20, color)
            arcade.draw_line(cell.x, cell.y, cell.x+1000*vec[0], cell.y+1000*vec[1], arcade.color.SPANISH_VIOLET)
            for neighbour in cell.links:
                if neighbour != cell:
                    arcade.draw_line(cell.x, cell.y, neighbour.x, neighbour.y, arcade.color.AZURE)

        #dx = sum([v[0] for v in org.vectors])
        #dy = sum([v[1] for v in org.vectors])
        #length = max([sqrt(v[0]**2 + v[1]**2) for v in org.vectors])


if __name__ == "__main__":
    moving_cells1 = [Muscle(150+30*cos(2*pi*i/4), 150+30*sin(2*pi*i/4)) for i in range(4)]
    moving_cells2 = [Muscle(450+50*cos(2*pi*i/3), 450+50*sin(2*pi*i/3)) for i in range(3)]

    for ind, cell1 in enumerate(moving_cells1):
        cell1.add_links(moving_cells1[:ind]+moving_cells1[ind+1:])
    for ind, cell2 in enumerate(moving_cells2):
        cell2.add_links(moving_cells2[:ind]+moving_cells2[ind+1:])

    org1 = Organism(moving_cells1)
    org2 = Organism(moving_cells2)
    on_draw.organisms = [org1, org2]
    on_draw.colors = [arcade.color.AMBER, arcade.color.AMETHYST]
    
    #print(org.get_average_coord())
    arcade.set_background_color(arcade.color.WHITE)
    arcade.schedule(on_draw, 1 / 80)
    arcade.run()
