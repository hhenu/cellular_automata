"""
Trying to create a smoothlife simulation. The simulation is shown live and
an animation is made of it (.gif format)..

Inspired by Tsoding:
https://www.youtube.com/playlist?list=PLpM-Dvs8t0VZxTsk3uYIM34QhMgNQP06G

See also the original paper: https://arxiv.org/abs/1111.1567
"""

import pygame
import numpy as np

from PIL import Image
from numba import njit, prange
from timeit import default_timer
from typing import Any, Callable


def timer(f: Callable) -> Callable:
    """
    Decorator to measure the time a function took to run
    :param f:
    :return:
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = default_timer()
        rv = f(*args, **kwargs)
        end = default_timer()
        print(f"Function '{f.__name__}' took {end - start:.4f} sec")
        return rv

    return wrapper


@njit
def sigma(x: float, a: float, alpha: float) -> float:
    """
    :param x:
    :param a:
    :param alpha:
    :return:
    """
    return 1 / (1 + np.exp(-(x - a) * 4 / alpha))


@njit
def sigma_n(x: float, a: float, b: float, alpha: float) -> float:
    """
    :param x:
    :param a:
    :param b:
    :param alpha:
    :return:
    """
    return sigma(x=x, a=a, alpha=alpha) * (1 - sigma(x=x, a=b, alpha=alpha))


@njit
def sigma_m(x: float, y: float, m: float, alpha: float) -> float:
    """
    :param x:
    :param y:
    :param m:
    :param alpha:
    :return:
    """
    s1 = 1 - sigma(x=m, a=.5, alpha=alpha)
    return x * s1 + y * sigma(x=m, a=.5, alpha=alpha)


@njit
def s(n: float, m: float, b1: float, b2: float, d1: float, d2: float,
      alpha_m: float, alpha_n: float) -> float:
    """
    :param n:
    :param m:
    :param b1:
    :param b2:
    :param d1:
    :param d2:
    :param alpha_m:
    :param alpha_n:
    :return:
    """
    a = sigma_m(x=b1, y=d1, m=m, alpha=alpha_m)
    b = sigma_m(x=b2, y=d2, m=m, alpha=alpha_m)
    return sigma_n(x=n, a=a, b=b, alpha=alpha_n)


@njit(parallel=True)
def calc_m_n(grid: np.ndarray, row: int, col: int, ra: int | float,
             ri: int | float, big_m: float, big_n: float) -> tuple[float, float]:
    """
    :param grid:
    :param row:
    :param col:
    :param ra:
    :param ri:
    :param big_m:
    :param big_n:
    :return:
    """
    rows, cols = grid.shape
    m, n = 0, 0
    for dy in prange(-ra, ra):
        y = (row + dy) % rows
        for dx in prange(-ra, ra):
            x = (col + dx) % cols
            if dx * dx + dy * dy <= ri * ri:
                m += grid[y, x]
            elif ri * ri <= dx * dx + dy * dy <= ra * ra:
                n += grid[y, x]

    return m / big_m, n / big_n


@njit
def _clamp(x: int | float, a: int | float, b: int | float) -> int | float:
    """
    :param x:
    :param a:
    :param b:
    :return:
    """
    if x < a:
        return a
    if x > b:
        return b
    return x


@njit(parallel=True)
def update(grid: np.ndarray, dt: float, ra: int | float, ri: int | float,
           big_m: float, big_n: float, b1: float, b2: float, d1: float,
           d2: float, alpha_m: float, alpha_n: float) -> np.ndarray:
    """
    :param grid:
    :param dt:
    :param ra:
    :param ri:
    :param big_m:
    :param big_n:
    :param b1:
    :param b2:
    :param d1:
    :param d2:
    :param alpha_m:
    :param alpha_n:
    :return:
    """
    rows, cols = grid.shape
    new_grid = np.zeros(shape=grid.shape)
    for y in prange(rows):
        for x in prange(cols):
            m, n = calc_m_n(grid=grid, row=y, col=x, ra=ra, ri=ri, big_m=big_m,
                            big_n=big_n)
            sv = s(n=n, m=m, b1=b1, b2=b2, d1=d1, d2=d2, alpha_m=alpha_m,
                   alpha_n=alpha_n)
            diff = 2 * sv - 1
            new_grid[y, x] = grid[y, x] + dt * diff
            new_grid[y, x] = _clamp(x=float(new_grid[y, x]), a=0, b=1)

    return new_grid


def _rgb_img(grid: np.ndarray) -> np.ndarray:
    """
    :param grid:
    :return:
    """
    grid = np.reshape(a=grid, newshape=(*grid.shape, 1))
    return np.repeat(grid, 3, axis=2) * 255


def main() -> None:
    # Initial settings
    width = 600  # Width of the grid
    height = 400  # Height of the grid
    ri = 4  # Radius of the inner circle
    ra_factor = 3  # Multiplier for the inner radius to get the outer radius
    ra = ri * ra_factor  # Outer radius
    b1 = 0.278  # Lower limit of the birth interval
    b2 = 0.365  # Upper limit of the birth interval
    d1 = 0.267  # Lower limit of the death interval
    d2 = 0.445  # Upper limit of the death interval
    alpha_m = 0.147  # Coefficient for the sigma functions
    alpha_n = 0.028  # Coefficient for the sigma functions
    dt = .1  # Timestep

    # Areas of the neighborhoods
    big_m = np.pi * ri * ri
    big_n = np.pi * ra * ra - big_m

    # Initialise a random grid
    grid = np.random.random(size=(height, width))

    # Setup pygame
    screen = pygame.display.set_mode(size=(width, height))
    pygame.display.set_caption(title="SmoothLife")

    # Show and save the animation
    images = []
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        grid = update(grid=grid, dt=dt, ra=ra, ri=ri, big_m=big_m,
                      big_n=big_n, b1=b1, b2=b2, d1=d1, d2=d2, alpha_m=alpha_m,
                      alpha_n=alpha_n)
        pygame.surfarray.blit_array(screen, _rgb_img(grid=grid.T))
        pygame.display.update()
        im = Image.fromarray(np.uint8(grid * 255), mode="L")
        images.append(im)

    images[0].save("smoothlife_anim.gif", save_all=True, append_images=images[1:],
                   fps=60)


if __name__ == "__main__":
    main()
