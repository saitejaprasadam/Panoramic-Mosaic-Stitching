import cv2 as opencv
import SIFT
import RANSAC
import sticher

boxes_image = opencv.imread("./project_images/Boxes.png")
image1 = opencv.imread("./project_images/Rainier1.png")
image2 = opencv.imread("./project_images/Rainier2.png")

boxes_details = SIFT.get_key_points(boxes_image, "1a")
image1_details = SIFT.get_key_points(image1, "1b")  # details is array 0 => keypoints and 1 => descriptor
image2_details = SIFT.get_key_points(image2, "1c")

matches = SIFT.fetch_matches(image1_details, image2_details)
matches_image = opencv.drawMatches(image1, image1_details[0], image2, image2_details[0], matches[:20], None, flags=2)
opencv.imwrite("./Output/2.png", matches_image)

homography, inv_homography, inlier_matches = RANSAC.ransac(matches, 4, 100, image1_details[0], image2_details[0], inlier_threshold=0.5)
inlier_image = opencv.drawMatches(image1, image1_details[0], image2, image2_details[0], inlier_matches, None, flags=2)
opencv.imwrite("./Output/3.png", inlier_image)

stitched_image = sticher.stitch(image1, image2, homography, inv_homography)
opencv.imwrite("./Output/4.png", stitched_image)
opencv.imshow("4.png", stitched_image)
opencv.waitKey(0)
print("Done")
