Count people in video (webcam or file) with YOLO

Download pre-trained YOLO v3 model from 
https://arxiv.org/abs/1804.02767 
and place in yolo-coco directory. Required files:
- coco.names
- yolov3.cfg
- yolov3.weights

App can be configured with ini file.

Usage:

python detector.py -c configfile

Example inifiles are provided