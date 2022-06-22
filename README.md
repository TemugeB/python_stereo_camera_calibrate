# What is this?
Stereo camera calibration script wriiten in python. Uses OpenCV primarily. 

# Why you want to stereo calibrate two cameras?
Allows you to obtain 3D points through triangulation from two camera views.

I wrote a [blog post](https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html) which turned out to be more popular than I expected. But it is a long blog post and some people just want the code. So here it is. Follow the instructions below and you should get working stereo camera calibration.

# Setup

Clone the repository to your PC. Then navigate to the folder in your terminal. 

**Install required packages**

This package uses ```python3.8```. Other version might result in issues. Only tested on Linux.

Other required packages are:
```
OpenCV
pyYAML
scipy -only if you want to triangulate. 
```
Install required packages:
```
pip3 install -r requirements.txt
```

**Calibration settings**

The camera calibration settings first need to be configured in the ```calibration_settings.yaml``` file. 
