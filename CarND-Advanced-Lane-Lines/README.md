## Advanced Lane Finding Project
Project Summary: 
Detect lane lines and curves in a video using *perspective transform* and *gradient threshholding* with OpenCV.
---

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./examples/undistorted_image_road.png "Road Distortion-Corrected"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_final_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

### [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is handled by the two functions `calibrateCamera()` and `undistortImage()`, line 9 through 30, in `ImageProcessor.py`.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

I use `undistortImage()` function in `ImageProcessor.py` to correct the distortion of each image. The input variables of this function are the results from the previous step, Camera Calibration, and the original image taken by the same camera.
![alt text][image2]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. This step is done by `imageTrhesholding()` in `ImageProcessor.py`. I used HLS color channel as well as Sobel_x and Sobel_y operators.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper()`, which appears in lines 92 through 96 in the file `ImageProcessor.py`. The `warper()` function takes as inputs an image (`img`), as well as the transform Matrix M. The transform Matrix was computed by the function `transformMatrix()` in `ImageProcessor.py` line 85 through line 90. I chose the hardcode the source and destination points in the following manner:

```python
src = np.float32(
	[[220, 700],
	 [530, 500],
	 [790, 500], 
	 [1100,700]])
dst = np.float32(
	[[220, 700],
	 [220, 450],
	 [1100, 450],
	 [1100, 700]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I fit my lane lines with a 2nd order polynomial using the window sliding methods. To do so, I created an object called `Lane` to keep track of the parameters and reduce the fitting time computations. Once warped image is prepared, a method called `Lane.LaneCalibration()` looks for lane curves and then saves or update the left and right curve polynomical coefficients.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

After I fit each lane to a polynomial of degree 2, the methods `computeCurvatureRadius()` and `updateVehicleCenter()`in `LaneFinder.py` compute the radii of curvature of the lanes and the position of the vehicle, respectively.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in `LaneFinder.py` in the function `visualizeLane()`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/5uLRnO21Kq8)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

First of all, the current way of estimating the vehicle's position with respect to center of the lane is not accurate enough as it uses the position of fitted curves and assumes that 920 pixel corresponds to 3.7 m.

We can also see from the result that computed value of radius of curvature fluctuates a lot, even though I use average of radii values from 10 consecutive frames. To obtain more reliable value, I can try to use reproduced (synthesized) pixel positions as demonstrated in the lecture, as opposed to detected lane pixels from raw image.

The current method would probably not so robust in rain and/or in different types of road (i.e. with other buildings around with narrow lanes).
