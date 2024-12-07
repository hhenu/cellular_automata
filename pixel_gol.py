"""
Cellular automata stuff
Conway"s Game of Life on a grid of pixels
"""

import pygame
import numpy as np

from numba import jit
from time import time
from copy import deepcopy
from numpy import ndarray


def timer(func):
    def wrapper(*args, **kwargs):
        start = time()
        rv = func(*args, **kwargs)
        print(f"Elapsed time {func.__name__}: {time() - start} s")
        return rv

    return wrapper


@jit(nopython=True)
def _find_live_neighbors(img_cpy: ndarray, row: int, col: int,
                         extended: bool = False,
                         gridmode: str = "moore",
                         donut: bool = True) -> int:
    """
    Finds the live neighbours around the pixel indicated by the row and
    col indices.

    This function uses numba to increase performance, and therefore
    it had to be defined outside of the class. The necessary instance
    parameters are passed as regular parameters from one of the instance
    methods.
    :param img_cpy: A copy of the image so that the original does not
        get modified during the process of searching neighbors
    :param row: Index of the row of the pixel under inspection
    :param col: Index of the column of the pixel under inspection
    :return: Number of live neighbors found
    """
    n_neighbors = 0
    lim_bot, lim_up = 1, 1
    if extended:
        lim_bot, lim_up = 2, 2

    if gridmode == "vonneumann" and not donut:
        for i in range(row - lim_bot, row + lim_up + 1):
            if i < 0 or i > (img_cpy.shape[0] - 1):
                continue
            if img_cpy[i][col] == 0 and i != row:
                n_neighbors += 1

        for j in range(col - lim_bot, col + lim_up + 1):
            if j < 0 or j > (img_cpy.shape[1] - 1):
                continue
            if img_cpy[row][j] == 0 and j != col:
                n_neighbors += 1

    if gridmode == "vonneumann" and donut:
        for i in range(row - lim_bot, row + lim_up + 1):
            i_ind = i % img_cpy.shape[0]
            if img_cpy[i_ind][col] == 0 and i_ind != row:
                n_neighbors += 1

        for j in range(col - lim_bot, col + lim_up + 1):
            j_ind = j % img_cpy.shape[1]
            if img_cpy[row][j_ind] == 0 and j_ind != col:
                n_neighbors += 1

    if gridmode == "moore" and not donut:
        for i in range(row - lim_bot, row + lim_up + 1):
            if i < 0 or i > (img_cpy.shape[0] - 1):
                continue
            for j in range(col - lim_bot, col + lim_up + 1):
                if i == row and j == col:
                    continue
                if j < 0 or j > (img_cpy.shape[1] - 1):
                    continue
                if img_cpy[i][j] == 0:
                    n_neighbors += 1

    if gridmode == "moore" and donut:
        for i in range(row - lim_bot, row + lim_up + 1):
            for j in range(col - lim_bot, col + lim_up + 1):
                if i == row and j == col:
                    continue
                i_ind = i % img_cpy.shape[0]
                j_ind = j % img_cpy.shape[1]
                if img_cpy[i_ind][j_ind] == 0:
                    n_neighbors += 1

    return n_neighbors


class PixelGameOfLife:
    """
    An object for Conway"s Game of Life type cellular automaton, where
    each pixel of an image is one cell. Uses numpy and opencv libraries.

    Supports two different styles of neighborhoods, the Moore (diagonals
    included) and the von Neumann (no diagonals) neighborhoods. Can handle
    the cells at the borders of the geometry in two different manners,
    either by considering the cells over the border to be dead, or by
    allowing the geometry to "wrap around" making it essentially a torus,
    aka a donut.
    """
    def __init__(self, w: int = 500, h: int = 500, fps: int = 20,
                 live_prob: float = 0.33, gridmode: str = "moore",
                 extended: bool = False, donut: bool = True,
                 use_numba: bool = False):
        """
        :param w: Width of the image
        :param h: Height of the image
        :param fps: Update rate, i.e. frames per second
        :param live_prob: Proportion of pixels set to be alive
        initially
        :param gridmode: Whether to use a Moore or von Neumann
        neighborhood
        :param extended: Whether to use the extended von Neumann
        neighborhood
        :param donut: Whether the geometry "wraps around" i.e.
        becomes a donut
        :param use_numba: Whether to use numba to increase
        performance
        """
        self.width = w
        self.height = h
        self.fps = fps
        self.live_prob = live_prob
        self.gridmode = gridmode
        self.extended = extended
        self.donut = donut
        self.use_numba = use_numba
        if self.gridmode != "moore" and self.gridmode != "vonneumann":
            self.gridmode = "moore"
            print("Using the default Moore neighborhood")
        self.img = np.ones((self.height, self.width))
        self.show_image()

    def init_img(self):
        """
        Initializes the image by setting the specified proportion
        of pixels alive, evenly and randomly distributed
        :return:
        """
        for i in range(self.img.shape[0]):
            for j in range(self.img.shape[1]):
                rand_num = np.random.rand()
                if rand_num <= self.live_prob:
                    self.img[i][j] = 0

    def find_live_neighbors(self, img_cpy: list, row: int, col: int) -> int:
        """
        Finds the live neighbours around the pixel indicated by the row and
        col indices
        :param img_cpy: A copy of the image so that the original does not
            get modified during the process of searching neighbors
        :param row: Index of the row of the pixel under inspection
        :param col: Index of the column of the pixel under inspection
        :return: Number of live neighbors found
        """
        n_neighbors = 0
        lim_bot, lim_up = 1, 1
        if self.extended:
            lim_bot, lim_up = 2, 2

        if self.gridmode == "vonneumann" and not self.donut:
            for i in range(row - lim_bot, row + lim_up + 1):
                if i < 0 or i > (len(img_cpy) - 1):
                    continue
                if img_cpy[i][col] == 0 and i != row:
                    n_neighbors += 1

            for j in range(col - lim_bot, col + lim_up + 1):
                if j < 0 or j > (len(img_cpy[0]) - 1):
                    continue
                if img_cpy[row][j] == 0 and j != col:
                    n_neighbors += 1

        if self.gridmode == "vonneumann" and self.donut:
            for i in range(row - lim_bot, row + lim_up + 1):
                i_ind = i % len(img_cpy)
                if img_cpy[i_ind][col] == 0 and i_ind != row:
                    n_neighbors += 1

            for j in range(col - lim_bot, col + lim_up + 1):
                j_ind = j % len(img_cpy[0])
                if img_cpy[row][j_ind] == 0 and j_ind != col:
                    n_neighbors += 1

        if self.gridmode == "moore" and not self.donut:
            for i in range(row - lim_bot, row + lim_up + 1):
                if i < 0 or i > (len(img_cpy) - 1):
                    continue
                for j in range(col - lim_bot, col + lim_up + 1):
                    if i == row and j == col:
                        continue
                    if j < 0 or j > (len(img_cpy[0]) - 1):
                        continue
                    if img_cpy[i][j] == 0:
                        n_neighbors += 1

        if self.gridmode == "moore" and self.donut:
            for i in range(row - lim_bot, row + lim_up + 1):
                for j in range(col - lim_bot, col + lim_up + 1):
                    if i == row and j == col:
                        continue
                    i_ind = i % len(img_cpy)
                    j_ind = j % len(img_cpy[i_ind])
                    if img_cpy[i_ind][j_ind] == 0:
                        n_neighbors += 1

        return n_neighbors

    @staticmethod
    @jit(nopython=True)
    def _update_img(img: ndarray, img_cpy: ndarray, extended: bool = False,
                    gridmode: str = "moore", donut: bool = True) -> ndarray:
        """
        Updates the image according to the amount of live neighbors
        each pixel had.

        This function uses numba to increase performance, thus it
        cannot take "self" as a parameter, and is therefore a staticmethod,
        and the necessary instance parameters are passed as normal
        parameters.
        :param img: The original image
        :param img_cpy: A copy of the image
        :return: Updated version of the image
        """
        for i in range(img_cpy.shape[0]):
            for j in range(img_cpy.shape[1]):
                live_neighbors = _find_live_neighbors(
                    img_cpy, i, j, extended, gridmode, donut
                )
                if img[i][j] == 0:
                    if live_neighbors < 2:
                        img[i][j] = 1
                    elif live_neighbors > 3:
                        img[i][j] = 1
                else:
                    if live_neighbors == 3:
                        img[i][j] = 0

        return img

    def update_img(self) -> None:
        """
        Updates the image according to the amount of live neighbors
        each pixel had. If numba is used to increase performance,
        this function calls the other function that is numba-combatible,
        i.e. a staticmethod.
        :return:
        """
        if not self.use_numba:
            img_cpy = list(list(i) for i in self.img)
            for i in range(len(img_cpy)):
                for j in range(len(img_cpy[i])):
                    live_neighbors = self.find_live_neighbors(img_cpy, i, j)
                    if self.img[i][j] == 0:
                        if live_neighbors < 2:
                            self.img[i][j] = 1
                        elif live_neighbors > 3:
                            self.img[i][j] = 1
                    else:
                        if live_neighbors == 3:
                            self.img[i][j] = 0
        else:
            img_cpy = deepcopy(self.img)
            self.img = self._update_img(self.img, img_cpy, self.extended,
                                        self.gridmode, self.donut)

    @staticmethod
    def _rgb_img(img: np.ndarray) -> np.ndarray:
        """
        :param img:
        :return:
        """
        img = np.reshape(a=img, newshape=(*img.shape, 1))
        return np.repeat(img, 3, axis=2) * 255

    def show_image(self) -> None:
        self.init_img()
        screen = pygame.display.set_mode(size=(self.width, self.height))
        pygame.display.set_caption(title="Game of Life")
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()

            pygame.surfarray.blit_array(screen, self._rgb_img(self.img.T))
            pygame.display.update()
            self.update_img()


def main():
    PixelGameOfLife(w=800//2, h=600//2, gridmode="moore",
                    fps=60, live_prob=0.5, use_numba=True)


if __name__ == "__main__":
    main()
