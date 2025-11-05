import numpy as np

def angle_3pts(a, b, c):
    """Angle at point b (in radians) for triangle a-b-c using vectors ba and bc."""
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cosang = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return np.arccos(cosang)

def rad2deg(r):
    return float(r * 180.0 / np.pi)

def safe_min(vals):
    vals = [v for v in vals if v is not None]
    return min(vals) if vals else None

def safe_mean(vals):
    vals = [v for v in vals if v is not None]
    return float(sum(vals)/len(vals)) if vals else None
