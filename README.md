# LeapDeepLearning
Gesture recognition using the convolutional neural networks and the Leap Motion controller

## Toolbox
This code uses the MatConvNet toolbox in Matlab for the convolutional neural network 
implementations, which comes from Oxford's VGG: http://www.vlfeat.org/matconvnet/

The toolbox can run on the CPU or GPU with the change of a single boolean in the code.

## Data
The data that the code runs on is located here: http://spiegel.cs.rit.edu/~hpb/LeapMotion/
It is approximately 1.5 GB and consists of over 9 thousand individual gestures collected 
from students, faculty, and staff at RIT.  The gestures collected consist of the following:

{'capE', 'CheckMark', 'e', 'F', 'Figure8', 'Swipe', 'Tap', 
'Grab', 'Release', 'Tap2', 'Wipe', 'Pinch' };

Further information on the collection of the data can be found in the DataCollector 
repository.
