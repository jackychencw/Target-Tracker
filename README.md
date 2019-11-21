# csc420-target-tracker

Members for this project: Yue Wang, Jue Wang, Ziqi Chen

Tasks for this project

1.	Detect faces in the video

2.	Perform face tracking by correctly associating a face detection in the previous frame to a face detection in the current frame.

3.	Train a classifier that matches each of the face with a face from the data set we provide, produce a softmax prediction, show the predicted name near the face

4.	For each frame add a new prediction toward the existing prediction list, classify the face as the majority of predictions

5.	User can provide a face that to be tracked, and if no such input is provided the application simply track all the faces appeared in the video

6.	If such input is provided, other than face detection we also perform a motion detection on the target and show a path for where the target has traveled

INSTRUCTION

To install requirments:
  using pip:
    pip install -R requirements.txt
  using conda:
    conda install --file conda-requirements.txt
