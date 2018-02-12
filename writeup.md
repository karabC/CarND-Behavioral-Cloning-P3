# **Behavioral Cloning** 
**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./cnn-architecture-624x890.png "Model Visualization"
[image2]: ./preprocessing.jpg "Preprocessing"
[image3]: ./left_right_flipped.jpg "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is based on the NVIDIA's model.
[NVIDIA model detail](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars)
![alt text][image1]
The model consists of a normalized layer, followed with 5 CNN and 3 Fully connected layers.


#### 2. Attempts to reduce overfitting in the model

To avoid overfitting, i tried to collect more data. Howevere, the sample data cannot get a good result with the self-generated data via keyboard. Then i decide to use the left and right camera instead.

Meanwhile, max pooling and dropout had been tried. The validation error is actually lower yet the car cannot pass the simulator eventually. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road by two correction by 0.24 and 0.27. These two indexes is indeed tuned by trial. The high correction of right is due to the crash on the bridge if that is set to 0.26 or below. However, left camera images are fine with the 0.23/0.24

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the NVIDIA one thought this model might be appropriate because that is very similar to the problem and the nature.

In order to gauge how well the model was working, I split my image into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. To avoid overfitting, i applied the max pooling and the droppout with the gridsearch approach. I come out with a low validation error model yet that still fail the simulatior run.

To combat the overfitting, I decide to levearage the steering angle data. However, the correction is simlilar to the hyperparameter. It takes time to tune. Originally, i added one more Fully connected layer at the end part of CNN with 200 nodes. However, after reconstructed more data, i found this layer is no longer useful and might even resulted in overfit too.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture

|      Layer      |               Description                |
| :-------------: | :--------------------------------------: |
|      Input      |            160 X 320 X3  RGB image             |
| Convolution 5x5 | 5x5 stride with 24 feature map |
|      RELU       |                                          |
| Convolution 5x5 | 5x5 stride with 36 feature map |
|      RELU       |                                          |
| Convolution 5x5 | 5x5 stride with 48 feature map |
|      RELU       |                                          |
| Fully connected |          Output:100          |
|      RELU       |                                          |
| Fully connected |          Output:50          |
|      RELU       |                                          |
| Fully connected |          Output:10          |
|      RELU       |                                          |
| Fully connected |          Output:1          |
|     MSE     |                                          |
|                 |                                          |
|                 |                                          |

#### 3. Creation of the Training Set & Training Process

I originally created two laps of data by using keyboard. But it is found that the peformance is really bad and data is quite different with the provided one.

I decide to use the provided data solely then because i find it is difficult to play the game using mouse instead of keyboard.

I finally randomly shuffled the data set and put 20% of the data into a validation set. There is a preprocessing step on chopping the data. see below

![alt text][image2]

Moreover, I have also leverage the left/right camera with steering. I have also flipped the image horizontalling to make the steering more stable. See below

![alt text][image3]

I used this enriched training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by the coverage speed. I used an adam optimizer so that manually training the learning rate wasn't necessary.
