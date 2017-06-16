
def clamp(v, low, high):
    v[v < low] = low
    v[v > high] = high
    return v