import numpy as np

def DoG_encoder(img):
    I = np.argsort(1 / img.flatten())  # Get indices of sorted latencies
    lat = np.sort(1 / img.flatten())  # Get sorted latencies
    I = np.delete(I, np.where(lat == np.inf))  # Remove infinite latencies indexes
    II = np.unravel_index(I, img.shape)  # Get the row, column and depth of the latencies in order
    t_step = np.ceil(np.arange(I.size) / ((I.size) / (15 - 6))).astype(np.uint8)
    II += (t_step,)
    spike_times = np.zeros((img.shape[0], img.shape[1], 15))
    
    spike_times[II] = 1
    
    return spike_times