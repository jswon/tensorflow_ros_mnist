# tensorflow_ros_mnist

This ROS package is made using the tensorflow provided by Google.

We used the handwritten digit recognition MNIST tutorial from tensorflow. 
It is a simple ROS package that I tried to get familiar with environment using tensorflow and ROS.
The main operation receives the camera image in real time and recognizes the number in the image

######
## Environment
- laptop : AilenWare17 R2
- VGA : Geforce 980M (Nvidia graphic driver version 367)
- OS : Ubuntu 14.04
- ROS version : Indigo
- tensorflow version : r0.12 (https://www.tensorflow.org/)
- cuda : 8.0                 (https://developer.nvidia.com/cuda-gpus)
- programming language : Python 2.7

#####

# Package configuration

## Train Package
 - Training Data set :
    -MNIST data set (http://yann.lecun.com/exdb/mnist/) 
    
 - Training Package (train.pkg) :
    - We modified the MNIST example provided by the tensor flow. 
    - It is a package that stores model parameters after training using MNIST dataset.

 
## Evaluation Package
 - Camera Package (VideoInput.pkg)
    - The image is input from the notebook's webcam. After simple image processing, publish the image.
    
 - Evaluation Package (Eval.pkg)
    - It is a package that evaluates input image and performs number recognition.
    
#####

Jong soon won.
