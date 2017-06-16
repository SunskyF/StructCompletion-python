import numpy as np

def scale_tform(H):
    uvTformScale = (H[:, 0] - H[:, 6] * H[:, 2]) * (H[:, 4] - H[:, 7] * H[:, 5]) - \
                   (H[:, 3] - H[:, 7] * H[:, 5]) * (H[:, 1] - H[:, 7] * H[:, 2])
    uvTformScale = np.sqrt(np.abs(uvTformScale))
    return uvTformScale