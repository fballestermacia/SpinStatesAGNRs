import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, eigvalsh
from TB_GNR import TB_HamArmchair_finite, TB_HamArmchair_infinite

def getGap(width,t,U):
    eigvalsinf = eigvalsh(TB_HamArmchair_infinite(width,U,t,0)) #the gap is always at k=0
    halfgap = np.min(np.abs(eigvalsinf))
    return halfgap


if __name__ == '__main__':
    
    t = -1
    U = 0
    
    widths = np.arange(100) + 2
    gaps = []
    quasiwidths = []
    ngaps = []
    
    
    for width in widths:
        gaps.append(getGap(width,t,U))
        print("\r Current width:{}".format(width))
    
    for i,gap in enumerate(gaps):
        if not np.isclose(gap,0):
            quasiwidths.append(widths[i])
            ngaps.append(widths[i]*gap)
        
    
    plt.figure()
    plt.subplot(121)
    plt.plot(widths,gaps,'k.-')
    plt.ylabel("Energy Gap (in units of t)")
    plt.xlabel("Width (number of atoms)")
    plt.text(75,0.3,"(a)",fontsize=20)
    plt.subplot(122)
    plt.plot(quasiwidths,ngaps,'k.-')
    plt.ylabel("Width * Energy Gap (in units of t)")
    plt.xlabel("Width (number of atoms)")
    plt.text(75,1.5,"(b)",fontsize=20)
    plt.show()