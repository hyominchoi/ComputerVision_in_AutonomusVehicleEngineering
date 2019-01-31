# Traffic Sign Recognition
Project summary: 
Build and train a traffic sign image classifier `LeNet()` with `TensorFlow`.

---
**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
 * Load the data set (see below for links to the project data set)
 * Explore, summarize and visualize the data set
 * Design, train and test a model architecture
 * Use the model to make predictions on new images
 * Analyze the softmax probabilities of the new images
 * Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/no_processing.jpg "Visualization"
[image2]: ./examples/y_factor.jpg "Grayscaling"
[image3]: ./examples/rgb2yuv.jpg "RGB to YUV Conversion"
[image4]: ./examples/children_28.jpg "Traffic Sign 1"
[image5]: ./examples/animal_31.jpg "Traffic Sign 2"
[image6]: ./examples/roundabout_40.jpg "Traffic Sign 3"
[image7]: ./examples/Pedestrians_27.jpg "Traffic Sign 4"
[image8]: ./examples/slippery_23.jpg "Traffic Sign 5"
[image10]: ./Dataset_label_counts.png "Data set label counts"
[image11]: ./Dataset_label_density.png "Data set label distribution"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Data Set Summary & Exploration

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 12630
* The size of test set is 4410
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

There are 43 classes and they are not uniformly distributed within each data set. For example, label number 1 and 2 appear more often than label 0. But we see that the three sets, train, validation, and test, share roughly the identical distribution.

![alt text][image10]
![alt text][image11]

### Design and Test a Model Architecture

#### 1. Preprocessing

As a first step, I decided to convert the images to YUV scale instead of RGB, then I normalized the Y factor and used raw and V coordinates as the [reference](yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) showed. Indeed, Y factor is the most dominant factor among 3 as we see in the following example, and so it makes sense to normalize Y factor.

Here is an example of a traffic sign image before and after RBG2YUV.

Original image
![alt text][image1]
YUV scale conversion
![alt text][image3]
Y factor from YUV scale
![alt text][image2]


#### 2. Model Architecture 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x30 	|
| tanh  				| Acivation										|
| Max pooling	      	| 2x2 stride,  outputs 14x14x15				    |
| Convolution 5x5       | 1x1 stride, valid padding, output 10x10x25    |
| tanh           	    | Activation  									|
| Max pooling	      	| 2x2 stride,  outputs 5x5x25				    |
| Fully connected		| 625 to output hidden layer 320        		|	
| tanh           	    | Activation  									|
| Fully connected		| Hidden layer, output 120                      |
| tanh           	    | Activation  									|
| logits        		| output 43 one hot encoded                     |
| Softmax				| Cross Entropy optimization   					|


#### 3. Model Training 

To train the model, I used the following parameters and estimators.

 * type of optimizer: AdamOptimizer
 * batch size : 128
 * number of epochs : 50
 * learning rate : 0.008
 * drop probability in dropbout: 0.5 


#### 4. Discussion

My final model results were:
 * training set accuracy of 1.0
 * validation set accuracy of 0.942 
 * test set accuracy of 0.965

I started with LeNet-5 architecture and tried several things including the model showed in the [reference](yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). However, I didn't achieve their accuracy level. I think there is some other points, such as parameter tuning, that I missed as the paper showed 0.98 accuracy on their test set.
The model architecture uses two flatten layers: flatten after the first max pooling and concatenate it to the Flatten layer after 2nd pooling. 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x30 	|
| tanh  				| Acivation										|
| Max pooling	      	| 2x2 stride,  outputs 14x14x15				    |
| Save Flatten 			| outputs 1x2940                                |
| Convolution 5x5       | 1x1 stride, valid padding, output 10x10x25    |
| tanh           	    | Activation  									|
| Max pooling	      	| 2x2 stride,  outputs 5x5x25				    |
| Fully connected		| tf.concat and 2940 + 625 to hidden, ouputs 320|
| tanh           	    | Activation  									|
| Fully connected		| Hidden layer, output 120                      |
| tanh           	    | Activation  									|
| logits        	    | output 43 one hot encoded                     |
| Softmax				| Cross Entropy optimization   					|

 
I replace RELU activation function by `tanh` as I normalized the data and so it makes more sense to choose an activation function of range [-1,1]. As a result, the accuracy has increased by about 0.015.

I increased the filter depth as well as the size of fully conntected layers because I used YUV, i.e. input image dimension was still 32x32x3. So using the original filter depth of 5 was not enough. I chose 15 and then 25 as filter depths.

I also added DROPOUT to prevent overfitting.

### Test a Model on New Images

#### 1. New Images
Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The last image might be difficult to classify because there is another cropped sign under the main sign. Moreover, the picture of a car and slippery road is relatively complicated looking.

#### 2. Discuss the model's predictions on these new traffic signs.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Children Crossing  	| Children Crossing  							| 
| Wild Animal   		| Wild Animal									|
| Roundabout			| Roundabout									|
| Pedestrians     		| Pedestrians					 				|
| Slippery Road			| Beware of ice/snow   							|



The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This looks low compared to the accuracy on the test set of 0.965, but we need to note that there are only 5 samples in this new set.

#### 3. Model Prediction in Detail


| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.99 				  	| Children Crossing  							| 
| 0.99          		| Wild Animal									|
| 0.99      			| Roundabout									|
| 0.97          		| Pedestrians					 				|
| 0.85      			| Beware of ice/snow   							|


For the last image, the model is relatively not sure that this is a ice/snow sign (probability of 0.85), and its second choice was Slippery sign with probability 0.078. In fact, the second choice is the correct label.


