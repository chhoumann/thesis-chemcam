from lib.reproduction import optimized_blending_ranges

"""
Get prediction y_full for each oxide for the full compositional range.
Use this prediction to find which model(sc are the best for each oxide.

If the prediction falls in a blending range, blend outputs of the two models.

Say we have y_full in the Low-Mid range. We blend the outputs of the Low and Mid models.
The final prediction, y_final, is a linear weighted sum of the two models' outputs.

w_mid = (y_full - blend_range_min) / (blend_range_max - blend_range_min)

"""


def f():
    return optimized_blending_ranges

