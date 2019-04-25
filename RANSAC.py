import cv2 as opencv
import numpy as np
import random

# TODO Verify with perspectiveTransform function
def project(point, homography):
    point = np.array([[point[0]], [point[1]], [1]])
    projected_dest = np.dot(homography, point)

    if projected_dest[2] is not 0:
        projected_dest = projected_dest / projected_dest[2]

    return np.array([projected_dest[0][0], projected_dest[1][0]])

def compute_inlier_count(homography, image1_keypoints, image2_keypoints, matches, inlier_threshold):

    src_pts = np.array([image1_keypoints[match.queryIdx].pt for match in matches], dtype=np.int)
    dest_pts = np.array([image2_keypoints[match.trainIdx].pt for match in matches], dtype=np.int)

    inlier_elements = []
    for index in range(0, len(src_pts)):
        src = src_pts[index]
        dest = dest_pts[index]
        projected_dest = project(src, homography)

        ssd = np.sqrt((np.array(projected_dest - dest, dtype=np.float32) ** 2).sum(0))
        if ssd < inlier_threshold:
            inlier_elements.append(index)

    return inlier_elements

def ransac_matches(inlier_matches, image1_keypoints, image2_keypoints):
    src_pts = np.array([image1_keypoints[match.queryIdx].pt for match in inlier_matches], dtype=np.int)
    dst_pts = np.array([image2_keypoints[match.trainIdx].pt for match in inlier_matches], dtype=np.int)
    homography, mask = opencv.findHomography(src_pts, dst_pts, method=0)
    return homography

def ransac(matches, num_matches, num_iterations, image1_keypoints, image2_keypoints, inlier_threshold):
    max_inlier_elements = []

    for index in range(0, num_iterations):
        random_4_matches = random.sample(matches, num_matches)
        src_pts = np.array([image1_keypoints[match.queryIdx].pt for match in random_4_matches], dtype=np.int)
        dst_pts = np.array([image2_keypoints[match.trainIdx].pt for match in random_4_matches], dtype=np.int)
        homography, mask = opencv.findHomography(src_pts, dst_pts, method=0)

        inlier_elements = compute_inlier_count(homography, image1_keypoints, image2_keypoints, matches, inlier_threshold)
        if len(inlier_elements) >= len(max_inlier_elements):
            max_inlier_elements = inlier_elements

    print("Max Inlier: " + str(len(max_inlier_elements)))

    inlier_matches = []
    for index in max_inlier_elements:
        inlier_matches.append(matches[index])

    homography = ransac_matches(inlier_matches, image1_keypoints, image2_keypoints)
    inv_homography = np.linalg.inv(homography)

    return homography, inv_homography, inlier_matches
