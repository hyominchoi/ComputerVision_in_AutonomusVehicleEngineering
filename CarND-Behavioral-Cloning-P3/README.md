# Behavioral Cloning
Project Summary: 
Build a pipeline that processes video data. Train a CNN model using `Keras` with GPU on AWS and successfully predict steering angles from new images.
---

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image2]: ./examples/center_2017_08_18_00_34_03_908.jpg "Center Image"
[image3]: ./examples/center_2017_08_18_00_33_53_077.jpg "Recovery Image"
[image4]: ./examples/center_2017_08_18_00_49_44_377.jpg "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/center_2017_08_18_00_34_08_343.jpg "Normal Image"
[image7]: ./examples/center_2017_08_18_00_34_08_343_flipped.jpg "Flipped Image"

### Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---

### Files Submitted to Udacity & Code Quality 

#### 1. List of all required files. They can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Functional Codes
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Check if code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.
Once parameters are modified appropriately, one can train and save the model by executing
```sh
python model.py
```

### Model Architecture and Training Strategy

#### 1. NVIDIA model architecture has been employed 

The model consists of a convolution neural network with 5x5 , 3x3 filter sizes and depths between 24 and 64 (model.py function run_model()). 

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.  

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py run_model()). 

The model was trained and validated on different data sets to ensure that the model was not overfitting, using sklearn.train_test_split. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 111).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I also added more data where the simulation tends to be off (for example, added more training data of bridge part of the track). I also added images obtained by driving in the reverse orientation.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the LeNet-5. However, this was not as accurate as I wanted it to be, so I took NVIDIA model as I thought this model was proven to be appropriate.

In the beginning, the validation mean square error was higher than the training error. So I obtained more training data using the simulator. Moreover, I added dropout layers to prevent overfitting. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track such as bridge part of the track. I added more data for certain spots. I also added data captured by left and right cameras, but the mean squred loss was not decreasing. So I decided to not to include those images. After certain point, adding more training data didn't help in terms of model performance, so I stopped adding more.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py line 93-108) consisted of 5 convlutional layers followed by 4 fully connected layers (last layer is the output layer). 


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![center image on the bridge][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to come back to center when it happens to go off the center. These images show what a recovery looks like starting from left or right:

![recovery from left side of the lane][image3]
![recovery from right side of the lane][image4]


To augment the data sat, I also flipped images and angles thinking that this would have the effect of gathering data as if the driving was done in the reverse orientation. For example, here is an image that has then been flipped:

![image][image6]
![flipped image][image7]


After the collection process, I had about 14K number of data points. With data augmentation, the size became 28K (or 52K if including left and right images).

I finally randomly shuffled the data set and put 20% of the data into a validation set. I cropped the image size to reduce the training time as well as increase the accuracy of the model.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5: I tried using 15 epochs in the beginning, but it didn't help to reduce mean squared errors with more than 5 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
