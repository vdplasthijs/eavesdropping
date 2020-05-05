import numpy as np

def angle_vecs(v1, v2):
    """Compute angle between two vectors with cosine similarity.

    Parameters
    ----------
    v1 : np.array
        vector 1.
    v2 : np.array
        vector 2.

    Returns
    -------
    deg: float
        angle in degrees .

    """
    assert v1.shape == v2.shape
    v1, v2 = np.squeeze(v1), np.squeeze(v2)
    tmp = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    rad = np.arccos(tmp)
    deg = rad * 360 / (2 * np.pi)
    return deg
