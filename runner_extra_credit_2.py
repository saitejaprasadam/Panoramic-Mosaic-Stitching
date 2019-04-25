import cv2 as opencv
import SIFT
import RANSAC
import sticher

images = [opencv.imread("./project_images/1.jpg"), opencv.imread("./project_images/2.jpg"),
          opencv.imread("./project_images/3.jpg"), opencv.imread("./project_images/4.jpg")]


def bla(image1, image2):
    current_image_details = SIFT.get_key_points(image1)
    next_image_details = SIFT.get_key_points(image2)
    matches = SIFT.fetch_matches(current_image_details, next_image_details)
    homography, inv_homography, inlier_matches = RANSAC.ransac(matches, 4, 500, current_image_details[0], next_image_details[0], inlier_threshold=0.5)
    return sticher.stitch(image1, image2, homography, inv_homography)

stitched = bla(images[1], images[2])
stitched = bla(stitched, images[0])
opencv.imwrite("./Output/custom_stitched.png", stitched)
opencv.waitKey(0)
print("Done Check for output/custom_stitched.png")
