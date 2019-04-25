# Panoramic Mosaic Stitching
Please adhere to your organization's rules for using this code. For [Concordia University](http://www.concordia.ca), these rules can be found [here](http://www.concordia.ca/students/academic-integrity/offences.html).

Stitching multiple images to form a panoramic image using OpenCV

1. Fetching matches
2. Performing RANSAC we get homography, inv_homography, inliers_matches.
	i. We take 4 random points and check how many inliers are present.
	ii. We perform this x times and note the max number of inliers.
	iii. With the max inliers we calculate the homography and inv_homography.

3. With the homography and inv_homography we gonna stitch the images.
	i. Calculate the dimension of the images to be projected on the projection surface.
	ii. Project image1 on projection plane.
	iii. Project the image2 on the projection plane.
