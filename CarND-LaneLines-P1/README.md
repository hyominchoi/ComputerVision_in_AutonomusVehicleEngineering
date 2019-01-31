# **Finding Lane Lines on the Road** 
---
Project summary: 
Using **Hough Transform** and **Canny Edge Detection** algorithm, we detect lane lines in any given image file.
---
Udacity project rubrics :
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report
---
### 1. Description of the pipeline and algorithm. 

My pipeline consisted of 7 steps. 

1) Apply color filters so that we only look at colors of interest such as yellow or white. 

2) Second, convert the images to grayscale.

3) Apply gaussian blur to smooth out the image.

4) Next, use canny edge detection algorithm to detect the edges of objects.

5) Define a 4-sided area of interest using polygon function.

6) Find out lines using Hough Transform.

7) Use linear regression to obtain smoothly connected lines.

In order to draw a single line on the left and right lanes, I used *numpy linear
regression function* ,`numpy.linalg.lstsq()`, before draw lines. I wrote `regression_line()` and `extrapolate_line()`.


### 2. Potential Shortcomings with my current pipeline

 - When the vehicle is on a more curved road, the current method would not work very accurately. 

 - Another shortcoming could be caused by changes in camera focal point.  If the camera's position is changed so that the focal point moves up, for example, then some parameters used in `pipeline()` need to be manually changed.


### 3. Suggestion of possible improvements to my current pipeline

1) To let pipeline handle curvy lanes, we should not use linear regression for `y = mx + c`. We can modify it to `y = ax^2 + bx + c` , for example. 

2) To prevent very abrupt changes, or discontinuity of projected lines, we can use line points from current and previous frames together and obtain smoothing effect.

3) Another potential improvement could be to build a function that detects camera's focal point using first few frames so that the relevant parameters get automatically adjusted whenever the position and/or angle of camera is changed.
