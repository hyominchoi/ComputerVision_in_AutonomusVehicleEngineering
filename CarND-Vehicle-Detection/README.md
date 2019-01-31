# Vehicle Detection Project
Project Summary: Build a pipepline that processes video data and detect cars.

---

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/HOG_example.jpg
[image3]: ./output_images/sliding_windows.jpg
[image4]: ./output_images/sliding_window.jpg
[image5]: ./output_images/bboxes_and_heat.png
[image6]: ./output_images/labels_map.png
[image7]: ./output_images/output_bboxes.png

### [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the code cell called 'Extracting features training sets' of the IPython notebook.  In this code sell, I used `extract_features` function in `lesson_functions.py`, which I took from the classroom code file and then made small changes on.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I didn't want to increase `orientation` as it took a while to extract HOG features when using `orientation = 9` already.


#### 2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and realized that HOG parameters extractred from HLS imgae works better than HSV or RGB color channels. So I decided to use `color_space = HLS`, along with `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using HLS color histogram with `hist_bins = 32` along with HOG features with parameters given above. I decided not to use `spatial_histogram` as SVM model's accuracy was greater than 98% without the spatial histogram.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search 4 different sizes of windows on the lower half region of the image. I let `cells_per_steop = 2` because I figured out that the `cells_per_steop = 3`  was too coarse. To decide on scales, I looked at the sizes of cars and locations on sample images. This search method, `multiple_window_sizes_search()` can be found in `detector.py`.

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on four scales using HLS 3-channel HOG features plus histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a link to my video result: https://youtu.be/5vDbt8N8XuQ


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I created a class `Detector` (in `detector.py`) to keep track of most recent records (`detector.heatmap_recent`, for exampe).

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. 

To reduce false positive, I created `filter_false_positive()` in `detector.py`. What this function does is take the most recent 5 to 10 frames to construct a heatmap. Once the heatmap is coonstructed, I apply `apply_threshold().`
I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six consecutive frames and their corresponding heatmaps:
![alt text][image5]


### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

First of all, this assumes a particular configuration of the camera. It will fail if the camera position changes in a way that the slding windows do not cover the road.

It is also very slow. I'd like it to be faster by implmementing a function that extracts HOG features at once for multiple size of windows (or different scale), instead of computing it every time the scale changes.

It still has false positives, and to improve on this, I'd like to make the model more robust, either using more various training data set and/or using CNN instead of SVM. Also, I'd like to change more parameters and see if the detection gets better. For example, I will try different color channels.
