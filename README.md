# **Behavioral Cloning** 


---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./imgs/Image1_NVidiaArchitecture.PNG "NVIDIA Architecture"
[image2]: ./imgs/Image2_modelSummary1.PNG "Model Summary"
[image3]: ./imgs/Image3_modelSummary2.PNG "Model Summary"
[image4]: ./imgs/image4_center.jpg "Center Camera Image"
[image5]: ./imgs/image5_left.jpg "Left Camera Image"
[image6]: ./imgs/image6_right.jpg "Right Camera Image"
[image7]: ./imgs/notflipped.png "Not Flipped Image"
[image7]: ./imgs/flipped.png "Image"


### Rubric Points
#### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3x3 and 5x5 kernel sizes and depths between 24 and 64 (model.py lines 17-32) 

The model includes RELU layers to introduce nonlinearity (code line 20), and the data is normalized in the model using a Keras lambda layer (code line 18,21 etc). 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets generated to ensure that the model was not overfitting (code line 47-49). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

We can add dropout layers as well in order to reduce overfitting.

#### 3. Model parameter tuning

The model used an adam optimizer

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. get_next_image_files function in helper.py file choses an image from either of the 3 cameras. For the left camera, a steering coefficient of 0.25 was added to mimic to move back towards the center of the road. Similarly, for the right camera, 0.25 was subtracted.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall model architecture is based on [NVIDIA's End to End Learning for Self-Driving Cars paper](https://arxiv.org/pdf/1604.07316.pdf).

![alt text][image1]

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. To combat the overfitting, I ran the model for lower number of epochs.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, as explained earlier i randomly select images from either of the cameras with appropriate steering coefficients.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road although the model accuracy was not that great.

#### 2. Final Model Architecture

The final model architecture (model.py lines 15-42) consisted of a convolution neural network with the following layers and layer sizes.

![alt text][image2]
![alt text][image3]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image4]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to adapt to such situation while driving autonomously. These images show what a recovery looks like starting from :

![alt text][image5]
![alt text][image6]


Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would add some extra data. For example, here is an image that has then been flipped:

Before:
![alt text][image7]

After:
![alt text][image8]


Also, the images from the training and test set are cropped to remove the sky, trees and hood. I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 8. I used an adam optimizer.
