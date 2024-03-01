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
In this code, we have a class STDP_learning.  
The input S_pre_sz is use to decide the data size from previous layer.  
The input s_pre is the spikes from previous layer.  
The input w is the weight of the layer using STDP in last timestep.  
The maxval is the max value of the neurons which are at the same position but in different layer. Becuase in STDP it also fellow the rule "lateral inhibition" which only allow the most active neuron fire in the same postion.  
Inputs maxind1 maxind2 are the postion for the max neruon.  
The input stride is the stride in preivous convolution layer which is used to decide which neruon should be updated.  
Inputs a_plus a_minus are learning rate for the STDP.  
The output of STDP_learing  is weight after learning.
## The main challenge of this reproduction is:
1. Except for the conlvution layer is finished by Norse package. The other function or layers even though are also included in the Norse package, the layers and modules needed in the paper are total differently. Thus, I have to write them by my self.
2. The debug for this code is hard since each modules is written by myself not from the standard package, sometimes the data formation should be changed and is hard to fix the problem.
3. The paper just give the brief structure of their network. SOme details should be chosen by myself.
## What is current status and what can be improved:
1. Right now, my network can reach 67% accuracy at the best performance and the average accuracy is 65%.
2. What can be improved is that check the result of each layer and compare them with paper's result, so that I can increase the accuracy to close to the result from the paper.
