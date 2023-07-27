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
    width=7
    length = 20
    
    
    with open("dataUmagnetization\width7length20andTemperature0.001at18_07_2023-11.06.24.dat") as r:
        rlines = r.readlines()[2:]
        Us = coltopoint(rlines,0)
        ms = coltopoint(rlines,1)
        
    plt.figure()
    plt.title("w={},l={}".format(width, length))
    plt.scatter(Us,ms,c='k',s=10)
    plt.vlines(2*np.abs(t),0,np.max(ms),linestyles='--')
    
    plt.xlabel('U/t')
    plt.ylabel('$m_L$')
    
    plt.show()