import numpy as np



def filter_RemoveImpulseNoise(dataPrev2, dataPrev1, dataCurr, thresh):
    pDataIn = []
    pDataIn.append(dataPrev2)
    pDataIn.append(dataPrev1)
    pDataIn.append(dataCurr)

    backwardDiff = pDataIn[1] - pDataIn[0]
    forwardDiff = pDataIn[1] - pDataIn[2]

    x1 = 0
    x2 = 2
    y1 = pDataIn[0]
    y2 = pDataIn[2]
    x = 1

    if ((forwardDiff > thresh and backwardDiff > thresh) or (forwardDiff < -thresh and backwardDiff < -thresh)):
        y = y1 + ((x - x1) * (y2 - y1) / (x2 - x1))
    else:
        y = pDataIn[1]
        
    return y

def remove_noise(delta_phase):
    thresh= 0.8
    data_hasil= []
#     n = (delta_phase.shape[0])-3
    for i in range(len(delta_phase)-3):# mulai data index ke-3 # mulai data index ke-3 
        dataPrev2 = delta_phase[i]# data[0] -> data(1)
        dataPrev1 = delta_phase[i+1]# data[1] -> data(2)
        dataCurr  = delta_phase[i+2] # data[2]
    #data_prev1.append(dataPrev1)
        y = filter_RemoveImpulseNoise(dataPrev2, dataPrev1, dataCurr, thresh)
        data_hasil.append(y)
    # data_hasil_AN = np.array(data_hasil)   
    return data_hasil
