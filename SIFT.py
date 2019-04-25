import cv2 as opencv
from copy import deepcopy

def get_key_points(image, file_name=None):
    orb = opencv.ORB_create()
    image1_key_points, image1_descriptor = orb.detectAndCompute(image, None)

    if file_name is not None:
        opencv.imwrite("./Output/" + file_name + ".png", draw_key_points(image, image1_key_points))

    return [image1_key_points, image1_descriptor]

def fetch_matches(image1_details, image2_details):
    bf = opencv.BFMatcher(opencv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(image1_details[1], image2_details[1])
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def draw_key_points(image, corner_points):

    image = deepcopy(image)
    for corner_point in corner_points:
        x, y = corner_point.pt
        opencv.circle(image, (int(x), int(y)), 4, (0, 0, 255))

    return image