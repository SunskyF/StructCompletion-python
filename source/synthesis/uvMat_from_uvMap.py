import numpy as np

def uvMat_from_uvMap(map, uvPixInd):
    H, W, Ch = map.shape

    offset = np.array(range(Ch), dtype=np.uint32) * W * H
    uvPixInd = uvPixInd[..., None] + offset[None, ...]
    uvMat = map[np.unravel_index(uvPixInd, map.shape, order="F")].copy()
    return uvMat