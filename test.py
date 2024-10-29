import numpy as np
from scipy.linalg import hadamard

if __name__ == '__main__':
    
    a = hadamard(16)
    b = a.T / 16
    aa = a[:4,:]
    bb = aa.T / 16
    print(bb)
    print(b[:,:4] == bb)






