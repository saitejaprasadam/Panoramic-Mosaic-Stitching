import cv2 as opencv
import numpy as np
import RANSAC

#TODO Chudali
def calculate_stitched_image_resolution(image1, image2, inv_homography):
    image1_height, image1_width, x = image1.shape
    image2_height, image2_width, x = image2.shape

    top_left = RANSAC.project([0, 0], inv_homography)
    top_right = RANSAC.project([image2_width - 1, 0], inv_homography)
    bottom_left = RANSAC.project([0, image2_height - 1], inv_homography)
    bottom_right = RANSAC.project([image2_width - 1, image2_height - 1], inv_homography)

    pan_left = int(min(top_left[0], bottom_left[0], 0))
    pan_right = int(max(top_right[0], bottom_right[0], image1_width))
    width = pan_right - pan_left

    pan_top = int(min(top_left[1], top_right[1], 0))
    pan_bottom = int(max(bottom_left[1], bottom_right[1], image1_height))
    height = pan_bottom - pan_top

    #TODO Chudali
    offset = (-pan_left, -pan_top)

    min_x = int(min(top_left[0], bottom_left[0]))
    min_y = int(min(top_left[1], top_right[1]))
    min_xy = (abs(min_x), abs(min_y))

    stitched_image_resolution = (height, width, 3)
    return stitched_image_resolution, offset, min_xy

#TODO can we use warpPerspective
#TODO Chudali
def stitch(image1, image2, homography, inv_homography):

    stitched_image_resolution, offset, min_xy = calculate_stitched_image_resolution(image1, image2, inv_homography)
    image1_height, image1_width, x = image1.shape
    #print(offset)
    #print(min_xy)

    # TODO Chudali
    (offset_x, offset_y) = offset
    #(min_x, min_y) = min_xy
    translation = np.matrix([
        [1.0, 0.0, offset_x],
        [0, 1.0, offset_y],
        [0.0, 0.0, 1.0]
    ])

    #offset_x
    #offset_y
    inv_homography = translation * inv_homography
    #homography = translation * homography
    (resolution_x, resolution_y, z) = stitched_image_resolution

    stitched_image = np.zeros(shape=stitched_image_resolution, dtype=np.uint8)
    stitched_image = opencv.warpPerspective(image2, inv_homography, (resolution_y, resolution_x), stitched_image)
    stitched_image[offset_y:image1_height + offset_y, offset_x:image1_width + offset_x] = image1

    """
    image2_height, image2_width, x = image2.shape
    for y_index in range(0, resolution_y):
        for x_index in range(0, resolution_x):
            projected_dest = RANSAC.project((y_index - min_y, x_index - min_x), homography)
            y = int(projected_dest[1])
            x = int(projected_dest[0])

            if 0 <= x < image2_width and 0 <= y < image2_height:
                stitched_image[x_index, y_index] = image2[y, x]"""

    return stitched_image
