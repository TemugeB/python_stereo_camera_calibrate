# What is this?
Stereo camera calibration script wriiten in python. Uses OpenCV primarily. 

# Why you want to stereo calibrate two cameras?
Allows you to obtain 3D points through triangulation from two camera views.

I wrote a [blog post](https://temugeb.github.io/opencv/python/2021/02/02/stereo-camera-calibration-and-triangulation.html) which turned out to be more popular than I expected. But it is a long blog post and some people just want the code. So here it is. Follow the instructions below and you should get working stereo camera calibration.

# Setup

Clone the repository to your PC. Then navigate to the folder in your terminal. Also print out a calibration pattern. Make sure it is as flat as you can get it. Small warps in the calibration pattern results in very poor calibration. Also, the calibration pattern should be properly sized so that both cameras can see it clearly at the same time. Checkerboards can be generated [here](https://calib.io/pages/camera-calibration-pattern-generator).

**Install required packages**

This package uses ```python3.8```. Other version might result in issues. Only tested on Linux.

Other required packages are:
```
OpenCV
pyYAML
scipy #only if you want to triangulate. 
```
Install required packages:
```
pip3 install -r requirements.txt
```

**Calibration settings**

The camera calibration settings first need to be configured in the ```calibration_settings.yaml``` file. 

```camera0```: This is your primary camera. You can check available video devices on linux with ```ls /dev/video*```. You only need to put the device number.

```camera1```: This is the secondary camera. 

```frame_width``` and ```frame_height```: Camera calibration is tied with the image resolution. Once this is set, your calibration result can only be used with this resolution. Also, both cameras have to have the exact same ```width``` and ```height```. If your cameras don't support the same resolution, resize your image in opencv to be the same ```width``` and ```height```. This package does not check if your camera resolutions are the same and does not raise exception if they are not the same or not supported by your camera. It is up to you to make sure your cameras can support this resolution.

```mono_calibration_frames```: Number of frames to use to obtain intrinsic camera parameters. 10 should be good.

```stereo_calibration_frames```: Number of frames to use to obtain extrinsic camera parameters. 10 should be good.

```view_resize```: If you're using a single screen and can't see both cameras because the images are too big, then set this to 2. This will show a smaller video feed but the saved frames will still be in full resolution.

```checkerboard_box_size_scale```: This is the size calibration pattern box in real units. For example, my calibration pattern is 3.19cm per box.

```checkerboard_rows``` and ```checkerboard_columns```: Number of crosses in your checkerboard. Note that this is NOT the number of boxes in your checkerboard. Check the image below for how to input these values.
![image](https://user-images.githubusercontent.com/36071915/175003788-b2477a50-6d73-45e1-a037-a317269fa9c1.png)



