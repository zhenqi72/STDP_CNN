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
Input x is the images from dataset.
## Lateral inhibition code:
Lateral_inhibit.py  
In this file, it only contain class Lateral_inhibit. This class is use to  realize the lateral inhibition function in the SNN.  
The input x are spikes from current layer.  
Mask is used to filter the spikes to ensure the right spikes can be sent to next layer.  
The m is is the index which is used to decide how to get the mask.
The v is the neuron voltages inside.
The output is x after filting and mask.
## Max pooling layer in SNN form:
Max_pooling_snn.py  
Input x is the spikes from previous layers.
The mask is used to control the output of the maxpooling.  
Since for each neurons in maxpooling layer, it can only fire for once.
## STDP module code:
STDP_selfmade.py
## 
