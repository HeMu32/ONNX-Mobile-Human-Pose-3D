# ONNX-Mobile-Human-Pose-3D
Python scripts for performing 3D human pose estimation using the Mobile Human Pose model.

This forked repo have been tested under Windows 10.19045 with python libraries:

certifi            2025.1.31
charset-normalizer 3.4.1
colorama           0.4.6
contourpy          1.3.1
cycler             0.12.1
filelock           3.18.0
fonttools          4.56.0
fsspec             2025.3.0
gitdb              4.0.12
GitPython          3.1.44
idna               3.10
imread_from_url    0.1.3
Jinja2             3.1.6
kiwisolver         1.4.8
MarkupSafe         3.0.2
matplotlib         3.10.1
mpmath             1.3.0
networkx           3.4.2
numpy              2.1.1
onnx               1.17.0
opencv-python      4.11.0.86
packaging          24.2
pandas             2.2.3
pillow             11.1.0
pip                25.0.1
protobuf           6.30.1
psutil             7.0.0
py-cpuinfo         9.0.0
pyparsing          3.2.1
PySocks            1.7.1
python-dateutil    2.9.0.post0
pytz               2025.1
PyYAML             6.0.2
requests           2.32.3
scipy              1.15.2
seaborn            0.13.2
setuptools         77.0.3
six                1.17.0
smmap              5.0.2
sympy              1.13.1
thop               0.1.1-2209072238
torch              2.6.0
torchvision        0.21.0
tqdm               4.67.1
typing_extensions  4.12.2
tzdata             2025.1
ultralytics        8.3.94
ultralytics-thop   2.0.14
urllib3            2.3.0
wheel              0.45.1
yt-dlp             2025.2.19



![Mobile Human 3D Pose mation ONNX](https://github.com/ibaiGorordo/ONNX-Mobile-Human-Pose-3D/blob/main/doc/img/output.bmp)
*Original image for inference: (https://static2.diariovasco.com/www/pre2017/multimedia/noticias/201412/01/media/DF0N5391.jpg)*

### :exclamation::warning: Known issues

 * The models works well when the person is looking forward and without occlusions, it will start to fail as soon as the person is occluded.
 * The model is fast, but the 3D representation is slow due to matplotlib, this will be fixed. The 3d representation can be ommitted for faster inference by setting **draw_3dpose** to False

# Requirements

 * **OpenCV**, **imread-from-url**, **scipy**, **onnx** and **onnxruntime**.

# Installation
```
pip install -r requirements.txt
```

# ONNX model
The original models were converted to different formats (including .onnx) by [PINTO0309](https://github.com/PINTO0309), download the models from [his repository](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/156_MobileHumanPose/download_mobile_human_pose_working_well.sh) and save them into the **[models](https://github.com/ibaiGorordo/ONNX-Mobile-Human-Pose-3D/tree/main/models)** folder. 

 * YOLOv5s: (In the original repo) You will also need an object detector to first detect the people in the image. Download the model from the [model zoo](https://github.com/PINTO0309/PINTO_model_zoo/blob/main/059_yolov5/22_yolov5s_new/download.sh) and save the .onnx version into the **[models](https://github.com/ibaiGorordo/ONNX-Mobile-Human-Pose-3D/tree/main/models)** folder.
 * However here in this fork, code were modified to work with onnx models converted by the script (export.py) in the Ultralytics official **[Yolo v5](https://github.com/ultralytics/yolov5)** repo, as the model used in the original repo has been deleted. To obtian a model, see **[here](https://docs.ultralytics.com/yolov5/tutorials/model_export/)** in the official document. Place the exported onnx model in the models folder.

# Original model
The original model was taken from the [original repository](https://github.com/SangbumChoi/MobileHumanPose).
 
# Examples

 * **Image inference**:
 
 ```
 python imagePoseEstimation.py 
 ```
 
  * **Video inference**:
 
 ```
 python videoPoseEstimation.py
 ```
 
 * **Webcam inference**:
 
 ```
 python webcamPoseEstimation.py
 ```
 
# [Inference video Example](https://youtu.be/bgjKKbGp5uo) 
 ![Mobile Human 3D Pose mation ONNX](https://github.com/ibaiGorordo/ONNX-Mobile-Human-Pose-3D/blob/main/doc/img/Mobile%20Pose%20Estimation%20ONNX.gif)

# References:
* Mobile human pose model: https://github.com/SangbumChoi/MobileHumanPose
* PINTO0309's model zoo: https://github.com/PINTO0309/PINTO_model_zoo
* PINTO0309's model conversion tool: https://github.com/PINTO0309/openvino2tensorflow
* 3DMPPE_POSENET_RELEASE repository: https://github.com/mks0601/3DMPPE_POSENET_RELEASE
* Original YOLOv5 repository: https://github.com/ultralytics/yolov5
* Original paper: 
https://openaccess.thecvf.com/content/CVPR2021W/MAI/html/Choi_MobileHumanPose_Toward_Real-Time_3D_Human_Pose_Estimation_in_Mobile_Devices_CVPRW_2021_paper.html
 


