
eps = 1e-16

def trans_tform(uvTformV, d_in):
    d = d_in.T.copy()

    uvTform = uvTformV.copy()

    if len(d.shape) == 1:
        uvTform[:, 6] = uvTformV[:, 0] * d[0] + uvTformV[:, 3] * d[1] + uvTformV[:, 6]
        uvTform[:, 7] = uvTformV[:, 1] * d[0] + uvTformV[:, 4] * d[1] + uvTformV[:, 7]
        uvTform[:, 8] = uvTformV[:, 2] * d[0] + uvTformV[:, 5] * d[1] + uvTformV[:, 8]
    else:

        uvTform[:, 6] = uvTformV[:, 0] * d[:, 0] + uvTformV[:, 3] * d[:, 1] + uvTformV[:, 6]
        uvTform[:, 7] = uvTformV[:, 1] * d[:, 0] + uvTformV[:, 4] * d[:, 1] + uvTformV[:, 7]
        uvTform[:, 8] = uvTformV[:, 2] * d[:, 0] + uvTformV[:, 5] * d[:, 1] + uvTformV[:, 8]
    uvTform = uvTform / (uvTform[:, 8] + eps)[..., None]
    return uvTform
