#**Traffic Sign Recognition** 

##Hyukpyo Hong February 13, 2017
---

**Behavrioal Cloning Project**

The goals / steps of this project are the following  

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road


[//]: # (Image References)

[image1]: ./model.png "Model Visualization"
[image2]: ./cen.jpg "center"
[image3]: ./rec1.jpg "Recovery Image - before"
[image4]: ./rec2.jpg "Recovery Image - after"
[image5]: ./tranoriginal.png "Original image before translation"
[image6]: ./tran30px.png "After move 30px"
[image7]: ./tranoriginal2.png "Original image before translation"
[image8]: ./tran10px.png "After move 10px"
[image9]: ./flipbefore.jpg "Original image"
[image10]: ./flipafter "Flipped Image"
[image11]: ./histogramafter "Histogram After"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

####2. Submssion includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
####3. Submssion code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model arcthiecture has been employed

My model consists of a convolution neural network with 

* 3 filters, size of 5x5, depth of 24, 36, 48, and stride of 2x2 valid
* 2 filters, size of 2x2, depth of 64, and stride of 1x1 valid 

and, linked to fully connected layers. 
Also The model includes seven **ELU** layerss to introduce nonlinearity.

####2. Attempts to reduce overfitting in the model

The model contains three **dropout layers**(0.5) and one **L2 W_regularlizer** in order to reduce overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 181-186). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an **adam** optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road from center, left, and right camera. And I flip and translated to the right and left to get more various situation.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to follow previous model, and add more layer when neccessary.

My first step was to use a convolution neural network model similar to the Nvidia end-to-end. I thought this model might be appropriate because they already succeed in self-driving with this model.But, since there were no information about what kind of activation layer or dropout layer was used, so that I started with a few 'relu' layer.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Although my first model had show a low mean squared error on both the training set and validation set, the car of simulator off the road in short time.

To combat the overfitting, I modified the model a lot of time. Actually, MSE factors always shows good results but performance of simulator was not enough. During the test, I realized that relu activation is not so useful in this case, becasuse they cannot make negative values, although angle require negative and positive values. 

Then I tested with ELI activation which can produce negative values, and the car of simulator moved more distance. The problem was the car easily failed on curved lane, and zigzaged during its driving. So I augmented the training imageset to make more training set which focused on curved lane, and lowered the throttle value, since my laptop is quite old model, so it cannot process images quickly which resulted in zigzag movement.

At the end of the process, the vehicle is able to drive autonomously around the track one and two without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 25-48) consisted of a convolution neural network 

Here is a summary of the architecture.

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded several laps on track one using center lane driving. Here is an example image of center lane driving,

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to back to center from edge of the lane. These images show what a recovery looks like starting from right to center.

![alt text][image3]
![alt text][image4]

To augment the data set, I translated the image with openCV. If angle value of image is more than 0.1, I moved the image 10px toward right or left, and add +-0.15 on original value. Also, if angle value of image is more than 0.3 , I moved the image 30px toward right or left, and add +-0.35 on original value. Here are sample images

#####[Left:Origianl, Angle:0.15 / Right:Move to right 10px, Angle 0.3]#####

![alt text][image7]
![alt text][image8]

#####[Left:Origianl, Angle:0.31 / Right:Move to right 35px, Angle 0.66]#####

![alt text][image5]
![alt text][image6]

I also flipped images and angles thinking that this make my imageset more be balanced. For example, here is an image that has then been flipped:

#####[Left:Origianl, Angle:0.5 / Right:Flipped image, Angle -0.5]#####
![alt text][image9]
![alt text][image10]

After flip the imageset, the histogram of angle value became to below chart.
![alt text][image11]


After the collection process, I had 52,442 number of data points. Although I tried to preprocess or nomalize the images by mean/std and min/max, it was failed. So I trained without any preprocessing.

I finally randomly shuffled the data set and put 5% of the data into a validation set. After I sured that my model is stable, I trained again with 0.1% of validation set, to improve model accuracy.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 11, since more than or less than 11 makes driving fail. I used an adam optimizer so that manually training the learning rate wasn't necessary.