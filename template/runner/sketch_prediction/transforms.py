import torch
import numpy as np


class ExtendSketchFormat(object):
    """
    TODO
    """

    def __call__(self, sketch):
        return extend_sketch_format(sketch)


def extend_sketch_format(sketch):
    """
    Extend the format of a sketch.
    Each point of the returned sketch is in the following format:
    (dx, dy, p1, p2, p3)
    where dx and dy are the relatives coordinates to the current point
    p1 indicates that the pen is touching the paper and that a line will be drawn from the current point to the next point
    p2 indicates that the pen will be lifted from the paper after the current point, and no line will be drawn next
    p3 indicates that it is the last point.
    """
    max_size = 256

    extended_sketch = []
    n = len(sketch)
    pp = 0
    for i in range(n):
        point = sketch[i]
        dx, dy, p = point
        if i == n - 1:
            # last point of the drawing
            extended_point = [dx, dy, 1 - pp, p, 1]
        else:
            # not the last point
            extended_point = [dx, dy, 1 - pp, p, 0]
        extended_sketch.append(extended_point)
        pp = p
    for i in range(max_size - n):
        extended_sketch.append([0, 0, 0, 0, 1])
    return extended_sketch


class SketchToTensor(object):
    """
    TODO
    """

    def __call__(self, sketch):
        return sketch_to_tensor(sketch)


def sketch_to_tensor(sketch):
    """
    TODO
    """
    sketch_array = np.array(sketch)
    sketch_tensor = torch.from_numpy(sketch_array)
    return sketch_tensor


class ScaleSketch(object):
    """
    TODO
    """

    def __call__(self, sketch, scale_factor=0.02323534459430452):
        return scale_sketch(sketch, scale_factor)


def scale_sketch(sketch, scale_factor):
    scaled_sketch = []
    n = len(sketch)
    point_size = len(sketch[0])
    for i in range(n):
        point = [float(sketch[i][0]) * scale_factor, float(sketch[i][1]) * scale_factor]
        for j in range(2, point_size):
            point.append(sketch[i][j])
        scaled_sketch.append(point)
    return scaled_sketch
