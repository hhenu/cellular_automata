"""
Cellular automata stuff
Conway"s Game of Life with larger cells
The "game" object itself
"""

import pygame
import random

from copy import deepcopy

# Define some constant variables, i.e. colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (50, 200, 50)
PURPLE = (200, 0, 255)


class GameOfLife:
    """
    An object for Conway's Game of Life type cellular automaton.
    Requires pygame to run.

    Supports two different styles of neighborhoods, the Moore (diagonals
    included) and the von Neumann (no diagonals) neighborhoods. Can handle
    the cells at the borders of the geometry in two different manners,
    either by considering the cells over the border to be dead, or by
    allowing the geometry to "wrap around" making it essentially a torus,
    aka a donut.

    The initial configuration can be set by hand, randomly (with a given
    proportion of the cells to be set alive), or by loading an existing
    configuration from a textfile, if configurations have been saved.

    Size of the grid, and of the squares within, is adjustable.
    """

    def __init__(self, sq_size: int = 12, n_grid_x: int = 60,
                 bg_color: tuple = BLACK, line_color: tuple = WHITE,
                 cell_color: tuple = GREEN, n_grid_y: int = 60,
                 fps: int = 12, rand_mode: bool = False,
                 live_prob: float = 0.33, gridmode: str = "moore",
                 extended: bool = False, donut: bool = True):
        """
        :param sq_size: Size of the individual cells on the grid
        :param n_grid_x: Number of grids in the x-direction (horizontal)
        :param n_grid_y: Number of grids in the x-direction (vertical)
        :param bg_color: Background color
        :param line_color: Color of the gridlines
        :param cell_color: Color of the live cells
        :param fps: Update rate aka frames per second
        :param rand_mode: Whether the grid is randomly generated
        :param live_prob: Amount of cells to be alive if the grid is to
            be randomly generated
        :param gridmode: Whether to use a Moore or von Neumann neighborhood
        :param extended: Whether to use the extended von Neumann
            neighborhood, i.e. increasing the range to two cells
        :param donut: Whether the geometry "wraps around", i.e. becomes
            a donut
        """
        self.sq_size = sq_size
        self.n_grid_x = n_grid_x
        self.n_grid_y = n_grid_y
        self.bg_color = bg_color
        self.line_color = line_color
        self.cell_color = cell_color
        self.fps = fps
        self.rand_mode = rand_mode
        self.live_prob = live_prob
        self.grid = []
        self.width = self.sq_size * self.n_grid_x
        self.height = self.sq_size * self.n_grid_y
        self.window = None
        self.n_rows = self.height // self.sq_size
        self.n_cols = self.width // self.sq_size
        self.gridmode = gridmode
        if self.gridmode != "moore" and self.gridmode != "vonneumann":
            self.gridmode = "moore"
            print("Using the default Moore gridmode")
        self.extended = extended
        if self.extended and self.gridmode != "vonneumann":
            self.gridmode = "vonneumann"
            print("Extended neighborhood only available with von Neumann "
                  "gridmode. Gridmode set to von Neumann.")

        self.donut = donut
        self.btn_w = 100
        self.btn_h = 50
        pygame.font.init()
        self.main_menu()

    def draw_gridlines(self) -> None:
        """
        Draws the gridlines at equal spacing, in vertical and
        horizontal directions
        :return:
        """
        x, y = 0, 0
        # Horizontal lines
        for _ in range(self.n_rows):
            pygame.draw.line(self.window, self.line_color, (0, y),
                             (self.width, y))
            y += self.sq_size

        # Vertical lines
        for _ in range(self.n_cols):
            pygame.draw.line(self.window, self.line_color, (x, 0),
                             (x, self.height))
            x += self.sq_size

        pygame.display.update()

    def make_grid(self) -> None:
        """
        Creates the grid in the numerical form, i.e. a nested list
        of ones and zeros
        :return:
        """
        if not self.rand_mode:
            self.grid = [[0] * self.n_cols for _ in range(self.n_rows)]
        else:
            self.grid = [[random.choice([0, 1]) for _ in range(self.n_cols)]
                         for _ in range(self.n_rows)]

    @staticmethod
    def get_mouse_pos() -> tuple:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        return mouse_x, mouse_y

    def color_cell(self, color: tuple, x_loc: int, y_loc: int) -> None:
        """
        Gives the cell at the given location a color
        :param color: Color as RGB-value
        :param x_loc: x-coordinate of the top left corner of the cell
        :param y_loc: y-coordinate of the top left corner of the cell
        :return:
        """
        pygame.draw.rect(self.window, color, (x_loc, y_loc, self.sq_size - 1,
                                              self.sq_size - 1))

    def draw_init_grid(self) -> None:
        """
        Allows for manually picking the cells which are alive at the start
        using mouse buttons (space to start the run)
        :return:
        """
        while True:
            for _ in pygame.event.get():
                if pygame.mouse.get_pressed(3)[0]:
                    x, y = self.get_mouse_pos()
                    if 0 < x < self.width and 0 < y < self.height:
                        x_ind = x // self.sq_size
                        y_ind = y // self.sq_size
                        self.grid[y_ind][x_ind] = 1
                        x_c = x_ind * self.sq_size + 1
                        y_c = y_ind * self.sq_size + 1
                        self.color_cell(self.cell_color, x_c, y_c)
                        pygame.display.update()

                if pygame.mouse.get_pressed(3)[2]:
                    x, y = self.get_mouse_pos()
                    if 0 < x < self.width and 0 < y < self.height:
                        x_ind = x // self.sq_size
                        y_ind = y // self.sq_size
                        self.grid[y_ind][x_ind] = 0
                        x_c = x_ind * self.sq_size + 1
                        y_c = y_ind * self.sq_size + 1
                        self.color_cell(self.bg_color, x_c, y_c)
                        pygame.display.update()

            keys = pygame.key.get_pressed()
            if keys[pygame.K_SPACE]:
                break

    def save_grid_config(self) -> None:
        """
        Saves the grid to a textfile, where each nested list (row)
        is on a new row
        :return:
        """
        filename = "yeet.txt"
        with open(filename, "w") as f:
            for row in self.grid:
                ind = 0
                for val in row:
                    if ind < (len(row) - 1):
                        f.write(str(val) + ",")
                    else:
                        f.write(str(val) + "\n")
                    ind += 1

    def load_grid_config(self, filename: str) -> None:
        """
        Loads a grid configuration from a textfile
        :param filename:
        :return:
        """
        with open(filename, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break
                line = [int(i) for i in line.split(",")]
                self.grid.append(line)

    def draw_grid(self) -> None:
        """
        Draws the cells with the corresponding color according to their state
        :return:
        """
        for i in range(len(self.grid)):
            for j in range(len(self.grid[i])):
                x_c = j * self.sq_size + 1
                y_c = i * self.sq_size + 1
                color = self.cell_color if self.grid[i][j] == 1 else self.bg_color
                self.color_cell(color, x_c, y_c)

        self.draw_gridlines()
        pygame.display.update()

    def find_live_neighbors(self, grid_cpy: list, row: int, col: int) \
            -> int:
        """
        Finds the live neighbours around the cell indicated by the row and
        col indices
        :param grid_cpy: A copy of the grid so that the original does not
            get modified during the process of searching neighbors
        :param row: Index of the row of the cell under inspection
        :param col: Index of the column of the cell under inspection
        :return: Number of live neighbors found
        """
        n_neighbors = 0
        lim_bot, lim_up = 1, 1
        if self.extended:
            lim_bot, lim_up = 2, 2

        if self.gridmode == "vonneumann" and not self.donut:
            for i in range(row - lim_bot, row + lim_up + 1):
                if i < 0 or i > (len(grid_cpy) - 1):
                    continue
                if grid_cpy[i][col] == 1 and i != row:
                    n_neighbors += 1

            for j in range(col - lim_bot, col + lim_up + 1):
                if j < 0 or j > (len(grid_cpy[0]) - 1):
                    continue
                if grid_cpy[row][j] == 1 and j != col:
                    n_neighbors += 1

        if self.gridmode == "vonneumann" and self.donut:
            for i in range(row - lim_bot, row + lim_up + 1):
                i_ind = i % len(grid_cpy)
                if grid_cpy[i_ind][col] == 1 and i_ind != row:
                    n_neighbors += 1

            for j in range(col - lim_bot, col + lim_up + 1):
                j_ind = j % len(grid_cpy)
                if grid_cpy[row][j_ind] == 1 and j_ind != col:
                    n_neighbors += 1

        if self.gridmode == "moore" and not self.donut:
            for i in range(row - lim_bot, row + lim_up + 1):
                if i < 0 or i > (len(grid_cpy) - 1):
                    continue
                for j in range(col - lim_bot, col + lim_up + 1):
                    if i == row and j == col:
                        continue
                    if j < 0 or j > (len(grid_cpy[i]) - 1):
                        continue
                    if grid_cpy[i][j] == 1:
                        n_neighbors += 1

        if self.gridmode == "moore" and self.donut:
            for i in range(row - lim_bot, row + lim_up + 1):
                for j in range(col - lim_bot, col + lim_up + 1):
                    if i == row and j == col:
                        continue
                    i_ind = i % len(grid_cpy)
                    j_ind = j % len(grid_cpy[i_ind])
                    if grid_cpy[i_ind][j_ind] == 1:
                        n_neighbors += 1

        return n_neighbors

    def update_grid(self) -> None:
        """
        Updates the grid according to the amount of live neighbors each
        cell had
        :return:
        """
        grid_cpy = deepcopy(self.grid)
        for i in range(len(grid_cpy)):
            for j in range(len(grid_cpy[i])):
                live_neighbors = self.find_live_neighbors(grid_cpy, i, j)
                if self.grid[i][j] == 1:
                    if live_neighbors < 2:
                        self.grid[i][j] = 0
                    elif live_neighbors > 3:
                        self.grid[i][j] = 0
                else:
                    if live_neighbors == 3:
                        self.grid[i][j] = 1

    def draw_text_middle(self, text: str, size: int, color: tuple):
        """
        Draws text in the main menu to the middle of the screen
        :param text: Text to be written
        :param size: Font size
        :param color: Color of the text
        :return:
        """
        font = pygame.font.SysFont("comicsans", size, bold=True)
        label = font.render(text, True, color)
        self.window.blit(label, (self.width // 2 - (label.get_width() // 2),
                                 self.height // 2 - (label.get_height() // 2)))

    def mainloop(self):
        """
        Mainloop that runs the game
        :return:
        """
        clock = pygame.time.Clock()
        self.window.fill(self.bg_color)
        self.draw_gridlines()
        if not self.rand_mode and not self.grid:
            self.make_grid()
            self.draw_init_grid()
            self.save_grid_config()
        elif self.rand_mode and not self.grid:
            self.make_grid()
        run = True
        while run:
            clock.tick(self.fps)
            self.draw_grid()
            self.update_grid()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False

    def main_menu(self):
        self.window = pygame.display.set_mode((self.width, self.height))
        left = self.width / 2 - self.btn_w / 2
        top = self.height / 2 - self.btn_h / 2
        button = pygame.Rect(left, top, self.btn_w, self.btn_h)
        run = True
        while run:
            self.window.fill(WHITE)
            pygame.draw.rect(self.window, BLACK, button)
            pygame.display.update()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if button.collidepoint(event.pos):
                        self.mainloop()

        pygame.display.quit()


def main():
    _ = GameOfLife(sq_size=12, n_grid_x=200, n_grid_y=140, rand_mode=True,
                   cell_color=PURPLE, donut=True)


if __name__ == "__main__":
    main()
