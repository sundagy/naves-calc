from __future__ import annotations

import math
from dataclasses import dataclass

from src.models import CanopyInput, Point2D


@dataclass(frozen=True)
class CanopyPlan:
    width: float = 12.0
    left_depth: float = 10.0
    right_depth: float = 7.0

    @property
    def points(self) -> list[Point2D]:
        return [
            Point2D(0.0, 0.0),
            Point2D(self.width, 0.0),
            Point2D(self.width, self.right_depth),
            Point2D(0.0, self.left_depth),
        ]

    @property
    def area(self) -> float:
        return self.width * (self.left_depth + self.right_depth) / 2.0

    def depth(self, x: float) -> float:
        return self.left_depth + (self.right_depth - self.left_depth) * x / self.width

    def contains_strictly(self, x: float, y: float, eps: float = 1e-9) -> bool:
        return eps < x < self.width - eps and eps < y < self.depth(x) - eps

    def on_boundary(self, x: float, y: float, eps: float = 1e-7) -> bool:
        if abs(y) <= eps and -eps <= x <= self.width + eps:
            return True
        if abs(x) <= eps and -eps <= y <= self.left_depth + eps:
            return True
        if abs(x - self.width) <= eps and -eps <= y <= self.right_depth + eps:
            return True
        return abs(y - self.depth(x)) <= eps and -eps <= x <= self.width + eps


@dataclass(frozen=True)
class RoofPlane:
    a: float
    b: float
    c: float
    width: float
    right_depth: float

    def z(self, x: float, y: float) -> float:
        return self.a * x + self.b * y + self.c

    @property
    def rear_right_height(self) -> float:
        return self.z(self.width, self.right_depth)

    @property
    def slope_x(self) -> float:
        return self.a

    @property
    def slope_y(self) -> float:
        return self.b

    @property
    def max_angle_deg(self) -> float:
        return math.degrees(math.atan(math.hypot(self.a, self.b)))


def roof_plane_from_input(data: CanopyInput) -> RoofPlane:
    a = (data.front_right_height - data.front_left_height) / data.width
    b = (data.rear_left_height - data.front_left_height) / data.left_depth
    c = data.front_left_height
    return RoofPlane(a=a, b=b, c=c, width=data.width, right_depth=data.right_depth)


def lower_truss_positions(data: CanopyInput) -> list[float]:
    if data.lower_positions_x:
        return [float(x) for x in data.lower_positions_x]
    if data.lower_count < 2:
        return [0.0]
    return [data.width * i / (data.lower_count - 1) for i in range(data.lower_count)]


def upper_truss_positions(data: CanopyInput) -> list[float]:
    if data.upper_count < 2:
        return [0.0]
    return [i / (data.upper_count - 1) for i in range(data.upper_count)]

