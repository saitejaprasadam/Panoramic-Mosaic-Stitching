import cv2 as opencv
import SIFT
import RANSAC
import sticher

images = [opencv.imread("./project_images/Rainier1.png"), opencv.imread("./project_images/Rainier2.png"),
          opencv.imread("./project_images/Rainier3.png"), opencv.imread("./project_images/Rainier4.png"),
          opencv.imread("./project_images/Rainier5.png"), opencv.imread("./project_images/Rainier6.png")]

def bla(image1, image2):
    current_image_details = SIFT.get_key_points(image1)
    next_image_details = SIFT.get_key_points(image2)
    matches = SIFT.fetch_matches(current_image_details, next_image_details)
    homography, inv_homography, inlier_matches = RANSAC.ransac(matches, 4, 2000, current_image_details[0], next_image_details[0], inlier_threshold=0.5)
    return sticher.stitch(image1, image2, homography, inv_homography)

stitched56 = bla(images[4], images[5])
opencv.imshow("stitched56.png", stitched56)
opencv.waitKey(0)

stitched12 = bla(images[0], images[1])
opencv.imshow("stitched12.png", stitched12)
opencv.waitKey(0)

stitched1256 = bla(stitched12, stitched56)
opencv.imshow("stitched1256.png", stitched1256)
opencv.waitKey(0)

stitched34 = bla(images[2], images[3])
opencv.imshow("stitched34.png", stitched34)
opencv.waitKey(0)

stitched_all = bla(stitched1256, stitched34)
opencv.imshow("stitched_all.png", stitched_all)
opencv.imwrite("./Output/AllStitched.png", stitched_all)
opencv.waitKey(0)