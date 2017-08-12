#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./image/placeholder.png "Model Visualization"
[image2]: ./image/centersample1.jpg "centersample track one"
[image3]: ./image/centersample2.jpg "centersample track two"
[image4]: ./image/leftsample1.jpg "leftsample track one"
[image5]: ./image/leftsample2.jpg "leftsample track two"
[image6]: ./image/rightsample1.jpg "rightsample track one"
[image7]: ./image/rightsample2.jpg "rightsample track one"
[image8]: ./image/rawforflipp.jpg "raw Image"
[image9]: ./image/rawforflipped.jpg "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:

* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.html summarizing the results
* video_one.mp4 showing the model driving the car autonomously in track one 
* video_two.mp4 showing the model driving the car autonomously in track two 

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and 3x3 filter sizes ,the depths of them are  both 16 (model.py lines 82-85) 

The model includes RELU layers to introduce nonlinearity (code line 86,90,93,96,99), and the data is normalized in the model using a Keras lambda layer (code line 79). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 88,91,94,97,100). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 109-110). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 106).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of a center, right and left camera data.
At the beginning of my project,I only collect data from track one. though I was not driving out of the road,but after training,the model's performance is bad. After some trials,I found that my training data is not so good,the steering is not very smooth,at some time the steering angle has some vibration. and this induce the model can not learn the good driving behavoir.
So in order to avoid this problem,I drive the car slowly and finely, control it on the center of the road track as possible as I can.It is a very useful change.After this improvment,I only collect one lap data on track one,and the model can drive the car on track one successfully.But I hope my model can drive on two of this track,the two is more diffcult than track one,so I collect more data from this two track.the finall dataset I used contains two laps of each track. 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to imitate the some existing good models.

My first step was to use a convolution neural network model similar to the Lenet-5. I thought this model might be appropriate because it contains some convolution layers and pooling layers to extract the image features ,it also contains several fullyconected layers to regress the steering angle.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model.
at first,I add three dropout layers to the behind of the last three dense layers.this can decrease overfitting a little bit.
Then I add more dropout layers to the model and elevate the dropout ratio of each dropout layers.
The final scheme is that there are five dropout layers and dropout ratio is  0.15 for two,0.25 for other three.

The final step was to run the simulator to see how well the car was driving around track one and track two. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 77-103) consisted of a convolution neural network with the following layers and layer sizes:

* convolution layer,kernel 5*5,the depth is 16
* maxpooling layer,kernel 2*2
* convolution layer,kernel 3*3,the depth is 16
* maxpooling layer,kernel 2*2
* RELU layer
* Flatten layer
* Dropout layer ,the dropout ratio set to be 0.15
* Dense layer,the depth is 256
* RELU layer
* Dropout layer ,the dropout ratio set to be 0.15
* Dense layer,the depth is 128
* RELU layer
* Dropout layer ,the dropout ratio set to be 0.25
* Dense layer,the depth is 64
* RELU layer 
* Dropout layer ,the dropout ratio set to be 0.25
* Dense layer,the depth is 32
* RELU layer
* Dropout layer ,the dropout ratio set to be 0.25
* Dense layer,the depth is 16
* Dense layer,the output layer,the depth is 1

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]
![alt text][image3]

I also used images from the left side and right cameras on the car so that the vehicle would learn to how to recover from the roadside when it is out of the edege.These images show what a recovery looks like:

![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]

Then I repeated this process on track two in order to get more data points.

To augment the data set, I also flipped images and angles thinking that this would get more data and augment the variaty of dataset,this method can enhance the model's generalization. For example, here is an image that has then been flipped:

![alt text][image8]
![alt text][image9]

After the collection process, I had 115665 number of data points. I then preprocessed this data by a Lambda layer and a Cropping2D layer.The Lambda layer can tranfer each pixel value from the region between 0 and 255 to the region between -0.5 and 0.5. The Cropping2D layer can cut off the top and the bottom parts.then the model can only the focus on the importmant part of the image.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was about 5 as evidenced by discovering that after 5 epoch running the training loss can reach  0.0560,and the validation loss also reach 0.0628,it is a good output which means the model is not overfitting and also not underfitting. I used an adam optimizer so that manually training the learning rate wasn't necessary.


