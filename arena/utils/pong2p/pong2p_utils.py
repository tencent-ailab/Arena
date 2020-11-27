from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pygame

class Rect():
    def __init__(self, x, y, w, h):
        self._x = x
        self._y = y
        self._w = w
        self._h = h

    def set_right(self, right):
        self._x = right - self._w

    def set_left(self, left):
        self._x = left

    def set_top(self, top):
        self._y = top

    def set_bottom(self, bottom):
        self._y = bottom - self._h

    def set_centery(self, centery):
        self._y = centery - self._h / 2.0

    def set_pos(self, x, y):
        self._x = x
        self._y = y

    def add_x(self, dx):
        self._x += dx

    def add_y(self, dy):
        self._y += dy

    def add_pos(self, dx, dy):
        self._x += dx
        self._y += dy

    def x(self):
        return self._x

    def y(self):
        return self._y

    def top(self):
        return self._y

    def bottom(self):
        return self._y + self._h

    def left(self):
        return self._x

    def right(self):
        return self._x + self._w

    def width(self):
        return self._w

    def height(self):
        return self._h

    def centerx(self):
        return self._x + self._w / 2.0

    def centery(self):
        return self._y + self._h / 2.0

    def rect(self):
        return pygame.Rect(self._x, self._y, self._w, self._h)
