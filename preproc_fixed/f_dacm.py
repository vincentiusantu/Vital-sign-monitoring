import numpy as np 
def dacm(data):
    psi=[]
    for i in range(1,data.shape[-1]):
        I = data[i].real
        Q = data[i].imag
        I_2 = data[i-1].real
        Q_2 = data[i-1].imag
        phase = (I*(Q- Q_2))-((I-I_2)*Q)
        phase2 = phase/((I**2)+(Q**2))
        psi.append(phase2)
    return psi