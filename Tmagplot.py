import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, eigvalsh
from TB_GNR import TB_HamArmchair_finite, TB_HamArmchair_infinite
from scipy.optimize import curve_fit, fsolve
import warnings

warnings.filterwarnings('ignore')

def coltopoint(array,column):
    #array: lista con archivos.dat o .txt
    #column: indice de la columna a modificar
    x = np.array([float(i.replace(',', '.'))  
        for i in [(line.split()[column]) 
        for line in array]])
    return x


if __name__=='__main__':
    
    t=-1
    width=13
    length = 30
    
    
    with open("dataTMagnetization\width13length30at22_07_2023-14.20.53.dat") as r:
        rlines = r.readlines()[30:]
        Ts = coltopoint(rlines,0)
        ms = coltopoint(rlines,1)
        
    plt.figure()
    plt.title("w={},l={}".format(width, length))
    plt.scatter(Ts,ms,c='k',s=10)
    plt.xscale('log')
    #plt.vlines(1e-2,0,np.max(ms),linestyles='--')
    
    plt.xlabel('$K_b T/t$')
    plt.ylabel('$m_L$')
    
    plt.show()