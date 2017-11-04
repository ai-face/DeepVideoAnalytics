

import cv2
import numpy as np

# average positions of face points
TEMPLATE = np.float32([(0.224152,  0.2119465),
                       (0.75610125, 0.2119465),
                       (0.490127, 0.628106),
                       (0.254149, 0.780233),
                       (0.726104, 0.780233)])


def crop(img, left, top, right, bottom) :
    if len(img.shape) == 2:
        return img[top:bottom, left:right]
    elif len(img.shape) == 3:
        return img[top:bottom, left:right, :]
    else:
        return None


def alignFace5P(img, landmark, size):
    H = cv2.getAffineTransform(np.float32(landmark), size * TEMPLATE)
    thumbnail = cv2.warpAffine(img, H, (size, size))
    return thumbnail


def drawLandmark(img, xs, ys, size=2):
    result = img.copy()
    assert len(xs) == len(ys)
    # draw landmarks
    if img.shape[2] == 3:
        value = [255, 0, 0]
    elif img.shape[2] == 4:
        value = [255, 0, 0, 0]
    else:
        return

    for i in range(len(xs)):
        result[int(xs[i]) - size:int(xs[i]) + size, int(ys[i]) - size:int(ys[i]) + size] = [255, 0, 0]

    return result

