import numpy as np


def cubic_blend(
    value,
    left_bottom, 
    right_bottom, 
    left_top, 
    right_top, 
    main_ratio, 
    cross_ratio):
    left_ratio = 1-main_ratio
    right_ratio = main_ratio
    # print(
    #     f"{value}\n"
    #     f"+ {(1-cross_ratio)}({left_bottom} {left_ratio} + {right_bottom} {right_ratio})\n"
    #     f"+ {cross_ratio} ({left_top} * {left_ratio} + {right_top} {right_ratio})\n"
    # )
    return value \
        + (1-cross_ratio) * (left_bottom * left_ratio + right_bottom * right_ratio) \
        + cross_ratio * (left_top * left_ratio + right_top * right_ratio)


def cubic_blend_2d(
    value_x,
    value_y,
    left_bottom_x, 
    left_bottom_y, 
    right_bottom_x,
    right_bottom_y,
    left_top_x, 
    left_top_y, 
    right_top_x, 
    right_top_y, 
    x_ratio, 
    y_ratio):
    return (
        cubic_blend(value_x, left_bottom_x, right_bottom_x, left_top_x, right_top_x, x_ratio, y_ratio),
        cubic_blend(value_y, left_bottom_y, left_top_y, right_bottom_y, right_top_y, y_ratio, x_ratio)
    )