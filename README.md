
[//]: # (Image References)

[image1]: ./images/sample_dog_output.png "Sample Output"
[image2]: ./images/vgg16_model.png "VGG-16 Model Keras Layers"
[image3]: ./images/vgg16_model_draw.png "VGG16 Model Figure"


# Artificial Intelligence Engineer Nanodegree: Deep Learning Applications
## Convolutional Neural Networks
### Dog Breed Classifier

## Project Overview

Used Convolutional Neural Networks (CNN) to build a pipeline to process real-world, user-supplied images. Given an image of a dog, my algorithm will identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed.

![Sample Output][image1]
![Sample Output][image2]
![Sample Output][image3]

Along with exploring state-of-the-art CNN models for classification, I made important design decisions about the user experience for your app. My goal was to understand the challenges involved in piecing together a series of models designed to perform various tasks in a data processing pipeline. Each model has its strengths and weaknesses, and engineering a real-world application often involves solving many problems without a perfect answer. My imperfect solution will nonetheless create a fun user experience!

### Code

* `dog_app.ipynb` - Interactive Python Notebook.
* `report.html` - Report of Python Notebook.

## Getting Started

To get this code on your machine you can fork the repo or open a terminal and run this command.
```sh
$ git clone https://github.com/JonathanKSullivan/dog_breed_classifier.git
$ cd dog_breed_classifier
$ jupyter notebook dog_app.ipynb
```

### Prerequisites

This project requires **Python 3**:

##### Notes:

1. It is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python and load the environment included below.

### Installing
#### Mac OS X and Linux

1. Run `git clone https://github.com/udacity/aind2-dl.git; cd aind2-dl`
2. Run `conda env create -f requirements/aind-dl-mac-linux.yml`
3. Run `source activate aind-dl`
4. Run `KERAS_BACKEND=tensorflow python -c "from keras import backend"`

#### Windows

1. Run `git clone https://github.com/udacity/aind2-dl.git; cd aind2-dl`
2. Run `conda env create -f requirements/aind-dl-windows.yml`
3. Run `activate aind-dl`
4. Run `set KERAS_BACKEND=tensorflow`
5. Run `python -c "from keras import backend"`


## Built With
* [Anaconda](https://www.continuum.io/downloads) - The data science platform used


## Future Steps!

(Presented in no particular order ...)

#### (1) Augment the Training Data 

[Augmenting the training and/or validation set](https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html) might help improve model performance. 

#### (2) Turn your Algorithm into a Web App

Turn your code into a web app using [Flask](http://flask.pocoo.org/) or [web.py](http://webpy.org/docs/0.3/tutorial)!  

#### (3) Overlay Dog Ears on Detected Human Heads

Overlay a Snapchat-like filter with dog ears on detected human heads.  You can determine where to place the ears through the use of the OpenCV face detector, which returns a bounding box for the face.  If you would also like to overlay a dog nose filter, some nice tutorials for facial keypoints detection exist [here](https://www.kaggle.com/c/facial-keypoints-detection/details/deep-learning-tutorial).

#### (4) Add Functionality for Dog Mutts

Currently, if a dog appears 51% German Shephard and 49% poodle, only the German Shephard breed is returned.  The algorithm is currently guaranteed to fail for every mixed breed dog.  Of course, if a dog is predicted as 99.5% Labrador, it is still worthwhile to round this to 100% and return a single breed; so, you will have to find a nice balance.  

#### (5) Experiment with Multiple Dog/Human Detectors

Perform a systematic evaluation of various methods for detecting humans and dogs in images. Provide improved methodology for the `face_detector` and `dog_detector` functions.

## Authors
* **Udacity** - *Initial work* - [AIND-Isolation](https://github.com/udacity/AIND-Isolation)
* **Jonathan Sulivan**

## Acknowledgments
* Hackbright Academy
* Udacity
