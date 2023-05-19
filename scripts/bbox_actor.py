import numpy as np
from typing import Tuple, List


class BBoxActor:
    def __init__(
        self,
        bbox_center_pos: List[float],
        bbox_center_velocity: List[float],
        bbox_size: List[float],
        color: Tuple[int, int, int],
        canvas_size: Tuple[int, int],
        noise_std_velocity: List[float] = [3.0, 1.0],
    ):
        self._bbox_center_pos_init: List[float] = bbox_center_pos  # (x, y)
        self._bbox_center_pos: List[float] = bbox_center_pos  # (x, y)
        self._bbox_center_velocity: List[float] = bbox_center_velocity  # (vx, vy)
        self._bbox_size: List[float] = bbox_size  # (width, height)
        self._color = color  # (b, g, r)
        self._canvas_size: Tuple[int, int] = canvas_size  # (width, height)
        self._noise_std_veclocity = noise_std_velocity  # (\sigma_vx, \sigma_vy)

    def reset(self):
        self._bbox_center_pos = self._bbox_center_pos_init.copy()

    def update(self):
        self._bbox_center_pos[0] += self._bbox_center_velocity[0] + np.random.normal(0, self._noise_std_veclocity[0])
        self._bbox_center_pos[1] += self._bbox_center_velocity[1] + np.random.normal(0, self._noise_std_veclocity[1])

    def get_bbox_info_as_xyxy(self) -> Tuple[int, int, int, int]:
        xs = int(self._bbox_center_pos[0] - self._bbox_size[0] / 2)
        ys = int(self._bbox_center_pos[1] - self._bbox_size[1] / 2)
        xe = int(self._bbox_center_pos[0] + self._bbox_size[0] / 2)
        ye = int(self._bbox_center_pos[1] + self._bbox_size[1] / 2)
        return (max(xs, 0), max(ys, 0), min(xe, self._canvas_size[0] - 1), min(ye, self._canvas_size[1] - 1))

    @property
    def bbox_center_pos(self):
        return self._bbox_center_pos

    @property
    def bbox_center_pos_init(self):
        return self._bbox_center_pos_init

    @property
    def bbox_size(self):
        return self._bbox_size

    @property
    def color(self):
        return self._color

    @property
    def bbox_xyxy_init(self) -> Tuple[int, int, int, int]:
        xs = int(self._bbox_center_pos_init[0] - self._bbox_size[0] / 2)
        ys = int(self._bbox_center_pos_init[1] - self._bbox_size[1] / 2)
        xe = int(self._bbox_center_pos_init[0] + self._bbox_size[0] / 2)
        ye = int(self._bbox_center_pos_init[1] + self._bbox_size[1] / 2)
        return (max(xs, 0), max(ys, 0), min(xe, self._canvas_size[0] - 1), min(ye, self._canvas_size[1] - 1))
