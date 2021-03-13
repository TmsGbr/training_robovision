# Project Documentation - RoboCup Vision (Thomas Gebauer 376238)

## Introduction
In order to get a robot to play soccer autonomously, there are several key factors.
Apart from the basics such as being able to execute movements like walking and kicking (Motion) and knowing where it is standing on the field (Localization), the robot also needs to be able to share information with its teammates (Communication) and to use the information it has, in order to decide how to react (Behavior). 

But all these crucial elements wouldn't work, if it weren't for Vision. 
The robot needs to perceive the world first, before being able to act accordingly. Only then can all the other software modules listed above make decisions and take action based on the representation of the state of the game.

The *NAO* robot uses two cameras to understand its environment. The software then detects and differentiates the different components of the robots surroundings, e.g. the goal posts, the ball, the field lines and obstacles such as other players. 

Despite the detection working fine without the use of artificial intelligence, I was determined to take a more experimental approach and figure out if and how machine learning could improve the process.
That being the case I decided to utilize machine learning based on *Neural Networks* fed by image data.
Using *TensorFlow* I trained a selection of *Convolutional Neural Network* models based on simulation data to build a vision module based largely on machine learning. 

Since I worked on this project by myself, I heavily relied on programming a solid infrastructure and creating a dependable dataset for training.
Instead of single-handedly having to hard-code rulesets I ran several rapid training iterations using various architectures and models.


## Prerequisites

### Software
The project is developed on *Windows 10* with and for ```python=3.7``` with ```tensorflow==2.3.1```. Corresponding to the *TensorFlow* version *CUDA Toolkit v10.1* ([installation guide](https://docs.nvidia.com/cuda/archive/10.1/cuda-installation-guide-microsoft-windows/index.html)) and *CuDNN 7.6.5* ([download](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/7.6.5.32/Production/10.1_20191031/cudnn-10.1-windows10-x64-v7.6.5.32.zip)) is used for GPU support.

Please refer to [this](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#tensorflow-object-detection-api-installation) guide to download the *TensorFlow Model Garden* and install *Protobuf*, the *COCO API* and the *TensorFlow Object Detection API*. During the process a newer version of *TensorFlow* might be installed. If this is the case there will be compatibility issues with *CUDA*. You can manually downgrade.
```shell
pip install tensorflow==2.3.1
```
To label the images I used *labelImg* (```pip install labelImg```).

For the best workflow clone this repository into ```Tensorflow/workspace``` (```Tensorflow``` being the *TensorFlow Model Garden* folder).

Additionally you need *[SimSpark](https://github.com/BerlinUnited/SimSpark-SPL/)* and *JupyterLab*.

### Hardware
To cope with the computation heavy task of training the the object detection model I used a *NVIDIA GeForce RTX 2070 SUPER*.
Using a GPU rather than a CPU for the training cuts the time needed by a large portion and is the reason why this project was even possible for me to take on within this time frame, without using external computational power.

Using the provided, readily trained model(s) from this repository should also work on devices with less computational power.


## Usage
This section will describe how to use the provided model(s) for inference. For a guide on how to train your own model please follow [Project Flow](#project-flow).

Start ```jupyter lab``` in the ```scripts``` folder and open ```use_model.ipynb```.

Import *TensorFlow* and suppress logging:
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow logging (1)
import pathlib
import tensorflow as tf

tf.get_logger().setLevel('ERROR')           # Suppress TensorFlow logging (2)
```

For GPU support:
```python
# Enable GPU dynamic memory allocation
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

Change ```PATH_TO_MODEL_DIR``` if you want to use a different model.

```python
# change for other models - '../exported-models/robo6-3-1' is recommended
PATH_TO_MODEL_DIR = '../exported-models/robo6-3-1'

PATH_TO_LABELS = '../annotations/label_map.pbtxt'
```

Load the model:
```python
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

print('Loading model...', end='')
start_time = time.time()

# Load saved model and build the detection function
detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

end_time = time.time()
elapsed_time = end_time - start_time
print('Done! Took {} seconds'.format(elapsed_time))
```
```
Loading model...Done! Took 8.26250171661377 seconds
```

Import the labels:
```python
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)
```

To run the inference and view the live results you need to open *SimSpark* and place it on your main monitor with no other window being in front of it. I don't have any way of testing the functionality of the screen capture on Linux, so if this is required to work on any other system than Windows and doesn't, please contact me.
The following cell will start an infinite loop of capturing the *SimSpark* window, performing object detection on this image and displaying the results in the notebook. This can be stopped with a ```KeyboardInterrupt```.
```python
import numpy as np
import warnings
import ipywidgets as widgets
from IPython.display import display
from PIL import ImageGrab
from PIL import Image
import win32gui
import cv2
warnings.filterwarnings('ignore')   # Suppress Matplotlib warnings

live_view = widgets.Image()

def perform_inference(image):

    image_np = np.array(image)

    # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
    input_tensor = tf.convert_to_tensor(image_np)
    # The model expects a batch of images, so add an axis with `tf.newaxis`.
    input_tensor = input_tensor[tf.newaxis, ...]

    # input_tensor = np.expand_dims(image_np, 0)
    detections = detect_fn(input_tensor)

    # All outputs are batches tensors.
    # Convert to numpy arrays, and take index [0] to remove the batch dimension.
    # We're only interested in the first num_detections.
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                   for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
          image_np_with_detections,
          detections['detection_boxes'],
          detections['detection_classes'],
          detections['detection_scores'],
          category_index,
          use_normalized_coordinates=True,
          max_boxes_to_draw=13,
          min_score_thresh=.30,
          agnostic_mode=False)

    _, encoded_image = cv2.imencode('.png', image_np_with_detections[:, :, [2, 1, 0]])
    live_view.value = encoded_image.tobytes()


display(live_view)


try:
    while(True):
        hwnd = win32gui.FindWindow(None, r'SimSpark')
        dimensions = win32gui.GetWindowRect(hwnd)

        # the dimensions have a boarder around the window, we subtract that
        x1, y1, x2, y2 = dimensions

        x1 += 8
        y1 += 31
        x2 -= 8
        y2 -= 8

        dimensions = (x1, y1, x2, y2)

        # capture the screen 
        image = ImageGrab.grab(dimensions)
        
        perform_inference(image)


except KeyboardInterrupt:
    print('Stopped')
```

## Project Flow

### Preliminary Considerations
Being alone on this project I had to make some early cutbacks to make this project possible in the given timeframe. Had I included a ball and other robots, I would not have been able to generate enough images for training and label them. I would have needed to reposition these components inbetween taking images, which would have made this process slower by a huge factor. On top of that, the labeling would have been even more time-consuming.
I also didn't have the time or resources to train really large and complex models, if I wanted to have any chance to experiment with different hyperparameters.

*SimSpark* doesn't offer an interface to get an image from the robots perspective, but instead calculates what the robot can see and gives positional information (in relation to the robot) of these elements. For real world application we wouldn't have this pre-calculated data, but instead would need to extract this data from an image/video-feed. To simulate this with *SimSpark* we can place the free moving camera in a position where a robot could be standing and use a screen capture of the *SimSpark* window as if it was the image from a robots camera. This won't work for a live match in *SimSpark* but will give an idea on how this approach could work for real live matches without any pre-computed data.

### Development
#### Capturing Images
To generate a dataset for the model training we first need the images (window captures) from *SimSpark*.
Run ```jupyter lab``` from ```scripts``` and open```screen_capture.ipynb```.

The first two cells just do some imports and a definition.

```python
from PIL import ImageGrab
from PIL import Image
import win32gui
import ipywidgets as widgets
import io
from IPython.display import display, clear_output
import os
```
```python
def image_to_byte_array(image:Image):
  imgByteArr = io.BytesIO()
  image.save(imgByteArr, format='PNG')
  imgByteArr = imgByteArr.getvalue()
  return imgByteArr
```

For the third cell to work, you need to open *SimSpark* and place it on your main monitor with no other window being in front of it. I don't have any way of testing the functionality of the screen capture on Linux. If this is required to work on any other System than Windows and doesn't, please contact me.
```python
cap_button = widgets.Button(description='Capture Window', icon='camera')
id_widget = widgets.IntText(value=0, description='Next free ID (change if you are re-running this cell)')
camera_widget = widgets.Image()

h_box = widgets.HBox([cap_button, id_widget])
v_box = widgets.VBox([h_box, camera_widget])

if not os.path.exists('../images'):
    os.makedirs('../images')

def capture_and_save_window(_):
    hwnd = win32gui.FindWindow(None, r'SimSpark')
    dimensions = win32gui.GetWindowRect(hwnd)

    # the dimensions have a boarder around the window, we subtract that
    x1, y1, x2, y2 = dimensions

    x1 += 8
    y1 += 31
    x2 -= 8
    y2 -= 8

    dimensions = (x1, y1, x2, y2)

    # capture the screen 
    image = ImageGrab.grab(dimensions)
    # convert image for preview
    byte_image = image_to_byte_array(image)
    
    # display preview
    camera_widget.width = (x2 - x1) / 2
    camera_widget.height = (y2 - y1) / 2
    camera_widget.value = byte_image
    
    # save image
    image.save(f'../images/{id_widget.value}.jpg', 'JPEG')
    id_widget.value += 1

cap_button.on_click(capture_and_save_window)
    
display(v_box)
```

There is a *'Capture Window'* button to save the current display of *SimSpark* as an image and show you a preview of the image that was saved. If you already have previously captured images in the ```images``` directory you need to set the next free ID manually in the widget.

I captured a total of 350 images that way, 'walking' the camera systematically over the field, taking pictures in every direction on every 'standing' point.

#### Labeling the Images
To label the images I used *labelImg*.
```shell
pip install labelImg
```

There are a total of 13 different labels. No two objects have the same label, instead every object has a different label, depending on its position on the field. I hoped to encode information on where the robot is in the vision model that way, but that might have proven to be a mistake, but more on that later. The position names are relative to the position the camera spawns in, in *SimSpark*.

There are 6 corners:
- ```left_near_corner```
- ```left_far_corner```
- ```middle_near_corner```
- ```middle_far_corner```
- ```right_near_corner```
- ```right_far_corner```

4 goalposts:
- ```left_near_goal```
- ```left_far_goal```
- ```right_near_goal```
- ```right_far_goal```

3 points for orientation:
- ```left_point```
- ```middle_point```
- ```right_point```

Labeling all 350 images with an average of around 3-5 labels per image took me way longer than what I previously accounted for. I spent nearly 8 hours drawing boxes and selecting labels in *labelImg*.

#### Creating a Label Map
*TensorFlow* needs a label map to assign an integer value to every label. I provide a ```label_map.pbtxt``` in the ```annotations``` folder.
```
item {
    id: 1
    name: 'left_near_corner'
}

item {
    id: 2
    name: 'left_near_goal'
}

.
.
.
```

#### Partitioning
In the ```scripts``` folder there is a partitioning script i got from [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#partition-the-dataset). From the ```scripts``` folder run it with:
```shell
python partition_dataset.py -x -i [PATH_TO_IMAGES_FOLDER] -r 0.1
```

This will randomly split the dataset into 90% training data and 10% test data. These images and the corresponding labels will be stored in ```images/train``` and ```images/test```.

#### Transforming Images and Labels to TFRecords
*labelImg* outputs the labels in the *PascalVOC* format, but *TensorFlow* needs them in ```TFRecord``` format.
I got a script for that from [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#convert-xml-to-record) and saved it in the ```scripts``` folder. It needs ```pandas``` (```pip install pandas```).

```shell
# Create train data:
python generate_tfrecord.py -x [PATH_TO_IMAGES_FOLDER]/train -l [PATH_TO_ANNOTATIONS_FOLDER]/label_map.pbtxt -o [PATH_TO_ANNOTATIONS_FOLDER]/train.record

# Create test data:
python generate_tfrecord.py -x [PATH_TO_IMAGES_FOLDER]/test -l [PATH_TO_ANNOTATIONS_FOLDER]/label_map.pbtxt -o [PATH_TO_ANNOTATIONS_FOLDER]/test.record
```

#### Download Pre-Trained Model
I downloaded a couple of pre-trained models from the [TensorFlow 2 Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). The two models I trained with are ```SSD ResNet50 V1 FPN 640x640 (RetinaNet50)``` and ```SSD MobileNet V2 FPNLite 640x640```, the latter being the one I got the most satisfying results with.

#### Training Pipeline Configuration
For every model I trained, I copied the provided ```pipeline.config``` into a folder inside the ```models``` folder and changed the parameters on where to find data and checkpoints and depending on how I wanted to train some of the training hyperparameters. More on that in Exploration.

#### Training the Model
*TensorFlow* provides a script for the model training, which I copied into the root directory of the repository. Run it from there.
```shell
python model_main_tf2.py --model_dir=models/<model folder> --pipeline_config_path=models/<model folder>/pipeline.config
```

#### Monitoring the Training
I used *TensorBoard* to monitor the training progress.
```shell
tensorboard --logdir=models/<model folder>
```

#### Model Export
*TensorFlow* provides a script for model export. Run it from the root directory with
```shell
python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\models\<model folder>\pipeline.config --trained_checkpoint_dir .\models\<model folder>\ --output_directory .\exported-models\<exported model folder>
```
My exported models can be found in the ```exported-models``` folder.

#### Using the Model
Please refer to the Usage chapter above.

### Exploration
Due to my limited resources, especially time (since I was alone), I couldn't test a huge amount of different models and model and training configurations, but I did make some findings that helped me further increase the capability of the models and with more fine tuning would help me to further increase them if I had more time.
For example, for my first attempt I used a batch size of 1 (I had memory problems in some tests with larger batch sizes) and didn't put a reasonable upper limit for the maximum amount of objects in the image, which resulted in a model, that after 25000 steps of training still had relatively high ```totalLoss``` of just under 2 and that couldn't find the field markers reliably.
My final/best model on the other hand (```exported-models/robo6-3-1```) used a batch size of 8 and knew that there could be a maximum of 13 objects and only needed 6000 steps (each taking longer to compute though) to perform on a considerably better level. After those 6000 steps the model actually got worse again, indicating overfitting. This could even be seen in the live view, where in some position object got high confidence scores and were detected correctly. But when moving away from a position like this the accuracy would drop by a large chunk and some objects couldn't be detected at all anymore.


## Results
I got a model that could find many, but not all of the objects when moving around the field. Especially problematic is, that due to the positional encoded labels the model for example can't decide if a point is left_point, right_point or middle_point. This results in a performance that is partially shaky, but overall proves that the general approach with machine learning is promising. It needs refinement to be more reliable but that will come with more experience.

## Next Steps
Developing the infrastructure and then training a *Convolutional Neural Network* based on simulation data and testing it for its accuracy worked fine within the simulation.
However, utilizing the AI model for use outside of the simulation is a different story.

Before anything else, the model would have to be retrained with real world images, captured through the video feed of the *NAOs* two cameras. In theory that is uncomplicated. The reality however is that images in the simulation tend to have clearer and more defined colors and shapes/edges compared to images captured in the real world. Which is why performance may vary when using real world images.

On top of that there is the aspect of the high computational cost when using artificial intelligence. During my project I had the code run on my personal hardware. This did not test whether the *NAO* would be capable to run the code fast enough on its own hardware or not. 
Since the long-term aim is to establish a team of fully autonomous humanoid soccer players that wins a soccer game against the winner of the most recent WorldCup, it is crucial that the players detect, differentiate and react to their changing surroundings as quickly as possible.

Despite all these challenges and room for improvements, the results that I've gathered in the course of this project are a good representation of the capability of an AI based vision module.
