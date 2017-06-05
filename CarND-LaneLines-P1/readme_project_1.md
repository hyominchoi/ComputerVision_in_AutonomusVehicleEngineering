# **Finding Lane Lines on the Road** 


---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 7 steps. 

Apply color filters so that we only look at colors of interest such as yellow or white. 

Second, convert the images to grayscale.

Apply gaussian blur to smooth out the image.

Next, use canny edge detection algorithm to detect the edges of objects.

Define a 4-sided area of interest using polygon function.

Find out lines using Hough Transform.

Use linear regression to obtain smoothly connected lines.


In order to draw a single line on the left and right lanes, I used numpy linear
regression function before draw lines. To be more specific, I didn't use draw_lines(). I wrote regression_line() and extrapolate_line(), which are used in lieu of draw_lines(). 


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when the vehicle is on more curved and less straight road. 

Another shortcoming could be caused by changes in camera focal point. 
If the camera's position is changed so that the focal point moves up, for example, then some parameters used in pipeline() need to be manually changed.


### 3. Suggest possible improvements to your pipeline

To let pipeline handle curvy lanes, we should not use linear regression for y = mx + c. We can modify it to y = ax^2 + bx + c, for example. 

To prevent very abrupt changes, or discontinuity of projected lines, we can use line points from current and previous frames together and obtain smoothing effect.

Another potential improvement could be to build a function that detects camera's focal point using first few frames so that the relevant parameters get automatically adjusted whenever the position and/or angle of camera is changed.
