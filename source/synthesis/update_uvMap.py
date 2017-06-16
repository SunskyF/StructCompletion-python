import numpy as np

def update_uvMap(map, data, uvPixInd_in):
    uvPixInd = uvPixInd_in.copy()
    H, W, Ch = map.shape

    offset = np.array(range(Ch), dtype=np.uint32) * W * H

    uvPixInd = uvPixInd[..., None] + offset[None, ...]

    map[np.unravel_index(uvPixInd, map.shape, order="F")] = data

    return map