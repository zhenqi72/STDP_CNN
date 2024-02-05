import numpy as np

def STDP_learning_CPU(S_sz, s, w, K_STDP,  # Input arrays
                  maxval, maxind1, maxind2,  # Indices
                  stride, offset, a_minus, a_plus):  # Parameters
    for idx in range(S_sz[0]):
        for idy in range(S_sz[1]):
            for idz in range(S_sz[2]):

                if idx != maxind1[idz] or idy != maxind2[idz]:  # Check if this is the neuron we have to update (correct idx, idy for map idz)
                    continue

                for i in range(w.shape[3]):
                    if (idz != i and maxind1[idz] <= maxind1[i] + offset
                        and maxind1[idz] >= maxind1[i] - offset
                        and maxind2[idz] <= maxind2[i] + offset
                        and maxind2[idz] >= maxind2[i] - offset
                        and maxval[i] > maxval[idz]):
                        maxval[idz] = 0.

                # Weights STDP update
                if maxval[idz] > 0:
                    # Weights STDP update
                    input = np.zeros(w[:, :, :, idz].shape)
                    if idy*stride >= S_sz[1] - w.shape[1] and idx*stride >= S_sz[0] - w.shape[0]:
                        ss = s[idx * stride:, idy * stride:, :]
                        input[:ss.shape[0], :ss.shape[1], :] = ss
                    elif idy*stride >= S_sz[1] - w.shape[1]:
                        ss = s[idx * stride:idx * stride + w.shape[0], idy * stride:, :]
                        input[:, :ss.shape[1], :] = ss
                    elif idx*stride >= S_sz[0] - w.shape[0]:
                        ss = s[idx * stride:, idy * stride:idy * stride + w.shape[1], :]
                        input[:ss.shape[0], :, :] = ss
                    else:
                        input = s[idx * stride:idx*stride+w.shape[0], idy*stride:idy*stride+w.shape[1], :]
                    dw = input * a_minus * w[:, :, :, idz] * (1 - w[:, :, :, idz]) + \
                         input * a_plus * w[:, :, :, idz] * (1 - w[:, :, :, idz]) - \
                         a_minus * w[:, :, :, idz] * (1 - w[:, :, :, idz])
                    w[:, :, :, idz] += dw

                    # Turn off the STDP for lateral neurons of the activated neuron in all planes
                    for k in range(S_sz[2]):
                        j = 0 if idy - offset < 0 else idy - offset
                        while j <= (S_sz[1] - 1 if idy + offset > S_sz[1] - 1 else idy + offset):
                            i = 0 if idx - offset < 0 else idx - offset
                            while i <= (S_sz[0] - 1 if idx + offset > S_sz[0] - 1 else idx + offset):
                                K_STDP[i, j, k] = 0
                                i += 1
                            j += 1

                    # Turn off the STDP for all neurons in the plane of the activated neuron
                    for j in range(S_sz[1]):
                        for i in range(S_sz[0]):
                            K_STDP[i, j, idz] = 0
    return w, K_STDP