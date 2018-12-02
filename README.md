# **Behavioral Cloning**
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[//]: # (Image References)

[image1]: ./readme_images/cnn-architecture-624x890.png "Model Architecture"
[image2]: ./readme_images/sample_preprocessed.png "Preprocessed sample"
[image3]: ./readme_images/left_2018_11_16_19_02_25_682.jpg "Raw sample - left camera"
[image4]: ./readme_images/center_2018_11_16_19_02_25_682.jpg "Raw sample - center camera"
[image5]: ./readme_images/right_2018_11_16_19_02_25_682.jpg "Raw sample - right camera"

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

## Local project setup

Running on local machine:
Assumes properly installed, configured and up to date both git and python.

My specs for comparison:
* Nvidia GTX 970
* Intel Xeon E3-1270 V5 3,6GHz

Step 1:
* Clone this repository.
Step 2:
* Setup a virtual environment inside cloned project, activate it and then install required packages
  ```sh
  virtualenv venv
  venv\Scripts\activate
  pip install -r requirements.txt
  ```
  This will install tensorflow-gpu!
  Here you can find instructions for enabling your GPU -> [TensorFlow GPU support](https://www.tensorflow.org/install/gpu)

Step 3:
* Install simulator -> [Udacity simulator](https://github.com/udacity/self-driving-car-sim)

Step 4:
* Running jupyter: `jupyter notebook`
* Running simulator with 'Autonomous mode', run following command `python drive.py model.h5`,
  and launch the simulator.

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* P4.ipynb notebook containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model_5.h5 containing a trained convolution neural network on my samples
* model_6.h5 containing a trained convolution neural network on Udacity samples
* README.md summarizing the results
* run7.mp4 video recording driving autonomously (model_5)
* run8.mp4 video recording driving autonomously (model_6)

#### 2. Submission includes functional code
Using the Udacity provided simulator and drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model_6.h5
```

#### 3. Submission code is usable and readable

The P4.ipynb notebook contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. 

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

Model consists of Normalization layer,
Three convolution layers with 5x5 filters applied, with striding 2x2, ReLU activation functions
Two convolution layers with 3x3 filters applied, without striding, ReLU activation functions
Three fully-connected layers

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer.

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.
I collected the data from both tracks, running both straight and in reverse direction.
I also tried to introduce dropout layers but to my own surprise I didn't get anything much from it. So I decided to drop them in final model.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and driving the turns smoothly. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

Unfortunately my initial models didn't guide me into the right direction and the model failed to turn on very sharp turns, also the autonomus driving was very unstable.
Therfore I decided to stick with the model already tested, mentioned in preparation lessons.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.
To combat the overfitting, I supplied the model with enough data from both tracks, also augmenting it.
The final step was to run the simulator to see how well the car was driving around track one.
At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

Input layer: IN: 66x200x3

Normalization layer: IN: 66x200x3 OUT:66x200x3

Convolution layer (activation: ReLU, stride: 2x2, filter: 5x5): IN: 66x200x3 OUT: 31x98x24

Convolution layer (activation: ReLU, stride: 2x2, filter: 5x5): IN: 31x98x24 OUT: 14x47x36

Convolution layer (activation: ReLU, stride: 2x2, filter: 5x5): IN: 14x47x36 OUT: 5x22x48

Convolution layer (activation ReLU, filter 3x3): IN: 5x22x48 OUT: 3x20x64

Convolution layer (activation ReLU, filter 3x3): IN: 3x20x64 OUT: 1x18x64

Flattening layer: IN: 1x18x64 OUT: 1164

Fully-connected layer: IN: 1164 OUT: 100

Fully-connected layer: IN: 100  OUT: 50

Fully-connected layer: IN: 50   OUT: 10

Output layer: IN: 10 OUT: 1

Here is a visualization of the architecture:
![Model architecture][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two and more laps on both tracks using center lane driving, driving in both directions.

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn how to react when getting closer to the edge of the road.

To augment the data, I also flipped images horizontally.

After the collection process, I had 38572 of samples in training set (Udacity datasets) and around 150 000 of samples in training set when training on my datasets. I then preprocessed this data as mentione before by passing images through hlp.preprocess() function.

Here is an example of preprocessed image:
![Preprocessed image][image2]

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2-3 as evidenced in notebook.

```
Train on 38572 samples, validate on 9644 samples
Epoch 1/3
38572/38572 [==============================] - 26s 677us/step - loss: 0.0165 - val_loss: 0.0187
Epoch 2/3
38572/38572 [==============================] - 21s 550us/step - loss: 0.0131 - val_loss: 0.0186
Epoch 3/3
38572/38572 [==============================] - 21s 551us/step - loss: 0.0118 - val_loss: 0.0210
```

I used an adam optimizer so that manually training the learning rate wasn't necessary.

run7.mp4 - shows the car driving autonomusly with model trained on my own samples.

run8.mp4 - shows the car driving autonomusly with model trained on Udacity samples.

simulation.mp4 - shows the car driving autonomusly recorded with included script capture_screen.py
