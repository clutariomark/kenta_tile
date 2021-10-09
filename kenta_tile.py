import argparse
import sys
from pathlib import Path
from typing import List
from typing import Optional
import numpy as np
import laspy
import pandas as pd
import os
import glob
import subprocess
import code

def recursive_split(x_min, y_min, x_max, y_max, max_x_size, max_y_size):
    x_size = x_max - x_min
    y_size = y_max - y_min

    if x_size > max_x_size:
        left = recursive_split(x_min, y_min, x_min + (x_size // 2), y_max, max_x_size, max_y_size)
        right = recursive_split(x_min + (x_size // 2), y_min, x_max, y_max, max_x_size, max_y_size)
        return left + right

    elif y_size > max_y_size:
        up = recursive_split(x_min, y_min, x_max, y_min + (y_size // 2), max_x_size, max_y_size)
        down = recursive_split(x_min, y_min + (y_size // 2), x_max, y_max, max_x_size, max_y_size)
        return up + down

    else:
        return [(x_min, y_min, x_max, y_max)]

las = laspy.read('E:\SkaenX\hydroanalysis 20210513\default.laz')
points = las.points

# preparation for feeding into dataframe
X = np.expand_dims(points.X, axis=1)
Y = np.expand_dims(points.Y, axis=1)
Z = np.expand_dims(points.Z, axis=1)
C = np.expand_dims(points.classification, axis=1)

xyzc = np.hstack([X, Y, Z, C])

# check if the shape is N-by-4
print(xyzc.shape)

# to visualize the las data, dataframe variable was created
df_las = pd.DataFrame(xyzc, columns = ['x', 'y', 'z', 'classification'])
df_las.head(10)

with laspy.open('E:\SkaenX\hydroanalysis 20210513\default.laz') as f:
    print(f"point format:      {f.header.point_format}")
    print(f"number of points:  {f.header.point_count}")
    print(f"number of vlrs:    {len(f.header.vlrs)}")


input_filename = 'E:\SkaenX\hydroanalysis 20210513\default.laz'
iterate_num = 10**10

# specify buffer size
buffer = 5
tile_size = 100
file_path = Path('E:\SkaenX\kenta_tile\output_100')

with laspy.open(input_filename) as file:

    sub_bounds = recursive_split(
        file.header.x_min,
        file.header.y_min,
        file.header.x_max,
        file.header.y_max,
        tile_size,  # tile size 1
        tile_size   # tile size 2
    )

    writers: List[Optional[laspy.LasWriter]] = [None] * len(sub_bounds)

    try:
        count = 0
        for points in file.chunk_iterator(iterate_num):
            print(f"{count / file.header.point_count * 100}%")

            # for performance we need to use copy so that
            # the underlying arrays are contiguous
            x, y = points.x.copy(), points.y.copy()

            point_piped = 0

            for i, (x_min, y_min, x_max, y_max) in enumerate(sub_bounds):
                mask = (x >= (x_min - buffer)) & (x <= (x_max + buffer)) & \
                       (y >= (y_min - buffer)) & (y <= (y_max + buffer))

                if np.any(mask):
                    
                    if writers[i] is None:
                        output_path = file_path.joinpath('output_ground_%s.laz' % str(i).rjust(5, '0'))
                        print(output_path)
                        writers[i] = laspy.open(
                            output_path,
                            mode = 'w',
                            header = file.header
                        )

                    sub_points = points[mask]
                    writers[i].write_points(sub_points)

                point_piped += np.sum(mask)
                if point_piped == len(points):
                    break

            count += len(points)

        print(f"{count / file.header.point_count * 100}%")

    finally:
        for writer in writers:
            if writer is not None:
                writer.close()

code.interact(local=dict(globals(), **locals()))