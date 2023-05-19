import numpy as np
from typing import Dict, Tuple, Optional


class Tracklet:
    def __init__(self, uuid: str):
        self._uuid = uuid
        self._frame_indices = []
        self._bbox_dict: Dict[int, np.ndarray] = {}  # bbox information is stored as xyxy format
        self._typical_bbox_size: Optional[Tuple[float, float]] = None  # stored as (width, height)
        self._typical_bbox_center_velocity: Optional[Tuple[float, float]] = None  # stored as (velocity_x, velocity_y)

    def append_bbox(self, frame_id: int, bbox: np.ndarray):
        self._frame_indices.append(frame_id)
        self._bbox_dict[frame_id] = bbox

    def calculate_typical_bbox_size(self):
        bbox_width_list = []
        bbox_height_list = []
        for frame_id in self._frame_indices:
            xs, ys, xe, ye = self._bbox_dict[frame_id]
            bbox_width = xe - xs
            bbox_height = ye - ys
            bbox_width_list.append(bbox_width)
            bbox_height_list.append(bbox_height)
        self._typical_bbox_size = (np.median(bbox_width_list), np.median(bbox_height_list))

    def _get_bbox_center(self, frame_id: int) -> Tuple[float, float]:
        xs, ys, xe, ye = self._bbox_dict[frame_id]
        center_x = (xs + xe) / 2
        center_y = (ys + ye) / 2
        return center_x, center_y

    def calculate_typical_bbox_center_velocity(self):
        if len(self._frame_indices) < 2:
            self._typical_bbox_center_velocity = (0.0, 0.0)
            return

        sorted_indices = sorted(self._frame_indices)
        velocities_x = [0]
        velocities_y = [0]
        prev_center_x, prev_center_y = self._get_bbox_center(sorted_indices[0])
        frame_id_prev = sorted_indices[0]
        for frame_id in sorted_indices[1:]:
            center_x, center_y = self._get_bbox_center(frame_id)

            # assume frame_id is corresponding to timestamp like 0, 1, 2, ...
            velocity_x = (center_x - prev_center_x) / (frame_id - frame_id_prev)
            velocity_y = (center_y - prev_center_y) / (frame_id - frame_id_prev)
            velocities_x.append(velocity_x)
            velocities_y.append(velocity_y)
            prev_center_x, prev_center_y = center_x, center_y
            frame_id_prev = frame_id

        self._typical_bbox_center_velocity = (np.median(velocities_x), np.median(velocities_y))

    @property
    def frame_indices(self):
        return self._frame_indices

    @property
    def track_length(self) -> int:
        return len(self.frame_indices)

    @property
    def bbox_dict(self):
        return self._bbox_dict

    @property
    def uuid(self) -> str:
        return self._uuid

    @property
    def typical_bbox_size(self) -> Tuple[float, float]:
        # Return size information as (width, height)
        return self._typical_bbox_size

    @property
    def typical_bbox_center_velocity(self) -> Tuple[float, float]:
        # Return velocity information as (velocity_x, velocity_y)
        return self._typical_bbox_center_velocity
