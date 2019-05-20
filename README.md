# R$^3$CNN: Regression Guided by Relative Ranking Using Convolutional Neural Network for Facial Beauty Prediction

R^3CNN is a general CNN architecture to integrate the relative ranking of faces in terms of aesthetics to improve the performance of facial beauty prediction.

## Requirements
* Caffe (compiled with pycaffe)
* python
* numpy
* matplotlib
* skimage

## Data preparation
Our method is trained and verified on the SCUT-FBP5500 benchmark. You can download the SCUT-FBP5500 dataset through this link: https://github.com/HCIILAB/SCUT-FBP5500-Database-Release.
Create a new folder named 'faces' and put in the images of SCUT-FBP5500. The train and test files have been provided in our data folder.

## Training
The training code will be public later. 

## Cross validation
1. Download the trained ResNeXt-based R$^3$CNN caffemodel from:
link: https://pan.baidu.com/s/1YVwKrBZS4kpNWHTRs-9qTA  password: xcx7 (1.6GB)
2. Create a new folder named 'models', and put in the download models 
3. Modify the path to test folder, and run the python file:
’‘’ python ./test_forward.py '''

## Contact Us
For any questions, please feel free to contact Luojun Lin (linluojun2009@126.com) or Prof. Jin (eelwjin@scut.edu.cn).

## Copyright
This code is free to the academic community for research purpose only. For commercial purpose usage, please contact Dr. Lianwen Jin: lianwen.jin@gmail.com.
