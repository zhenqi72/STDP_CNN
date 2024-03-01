# What is inside this reporsitory:
This is the paper reproduction of paper "STDP-based spiking deep convolutional neural networks for object recognition"
# What is the target for this paper reproduction:
The main target for this reproduction is to learn SNN and creat a test model for homomorphic encryption by SNN.
# The structure of this code
## train snn and svm classifier code:
train_snn.py 
This is the main code which is used to train the snn and classifier.
## DoG fitler in SNN form code:
DoGFilter_spike.py 
In this code, it mainly contains two parts the class DoGFilter and DoG_encoder function. Inside the DoGFilter it contains the DoG filter and use the DoG_encoder function to conver the real numer to spikes.
## Lateral inhibition code:
Lateral_inhibit.py
## Max pooling layer in SNN form:
Max_pooling_snn.py
## STDP module code:
STDP_selfmade.py
## 
