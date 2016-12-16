# tensorflow_ros_mnist
## Summary
This ROS package is made using the tensorflow provided by Google.

We used the handwritten digit recognition MNIST tutorial from tensorflow. 
It is a simple ROS package that I tried to get familiar with environment using tensorflow and ROS.
The main operation receives the camera image in real time and recognizes the number in the image

######
## Environment
- webcam : logitech c920r 
- laptop : AilenWare17 R2
- VGA : Geforce 980M (Nvidia graphic driver version 367)
- OS : Ubuntu 14.04
- ROS version : Indigo
- tensorflow version : r0.12 (https://www.tensorflow.org/)
- cuda : 8.0                 (https://developer.nvidia.com/cuda-gpus)
- cudnn : 5
- Python 2.7
- opencv2 python

#####

# Package configuration 
## Train Package
 - Training Data set :
    -MNIST data set (http://yann.lecun.com/exdb/mnist/)    
    
 - Training Package (train.pkg/train.py) :
    - We modified the MNIST example provided by the tensor flow. 
    - It is a package that stores model parameters after training using MNIST dataset.

 
## Evaluation Package
 - Camera Package (Cam_image.pkg/Cam_image.py)
    - The image is input from the webcam. After simple image processing, publish the image.
    
 - Evaluation Package (eval.pkg/eval.py)
    - It is a package that evaluates input image and performs number recognition.
    
## Comment 
 - Each package is run after you add execute permissions to the * .py file.
    - $chmod +x [Package_name] [Package_File_name].py
    - $rosrun [Package_name] [Package_File_name].py
 
## Topic list
  - Image(name : "video")
  
#####

jong-soon won 
