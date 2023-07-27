import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, eigvalsh


def lorentzian(x,x0,g): #this is just for plotting
    return 1/np.pi*(0.5*g/((x-x0)**2+(0.5*g)**2))


def Tb_Haiku_Infinite(U,t,kx):
    UCnsites = 5*4+7*2 #34
    H0 = np.diag(U/2*np.ones(UCnsites, dtype="complex128"))
    infivel = True
    inseven = False
    infiver = False
    
    abovejump = np.zeros(UCnsites-1, dtype="complex128")
    for i in range(UCnsites):
        if infivel:
            if i%5 != 4:
                abovejump[i] = t
            elif i==9:
                infivel = False
                inseven = True
                infiver = False
        elif inseven:
            if (i-10)%7 != 6:
                abovejump[i] = t
            elif i==23:
                infivel = False
                inseven = False
                infiver = True
        elif infiver:
            if (i-24)%5 != 4:
                abovejump[i] = t
                
    sidejumpfive = np.zeros(UCnsites-5, dtype="complex128")
    sidejumpchange = np.zeros(UCnsites-6, dtype="complex128")
    sidejumpseven = np.zeros(UCnsites-7, dtype="complex128")
    
    infivel = True
    inseven = False
    infiver = False
    for i in range(UCnsites):
        if infivel:
            if i==9:
                infivel = False
                inseven = True
                infiver = False
                sidejumpchange[i]=t
            elif i>4:
                if ((i+i//5)%2)==0:
                    if i//5==1:
                        sidejumpchange[i]=t
                    else:
                        sidejumpfive[i]=t
            else:
                if ((i+i//5)%2)==1:
                    if i//5==1:
                        sidejumpchange[i]=t
                    else:
                        sidejumpfive[i]=t
            
        elif inseven:
            if i ==23:
                infivel = False
                inseven = False
                infiver = True
            elif i>16:
                if (((i-10)+1-(i-10)//7)%2) == 0:
                    if (i-10)//7 ==1:
                        sidejumpchange[i] = t
                    else:
                        sidejumpseven[i] = t
            else:
                if (((i-10)+(i-10)//7)%2) == 0:
                    sidejumpseven[i] = t
            
        elif infiver:
            if i==25 or i==27:
                sidejumpfive[i] = t
    
    periodjump = np.zeros(UCnsites-29,dtype='complex128')
    for i in range(len(periodjump)):
        if i%2 == 0:
            periodjump[i] = t*np.exp(-1j*kx)
    
    H0 += np.diag(abovejump,k=1) + np.diag(sidejumpfive,k=5) + np.diag(sidejumpchange,k=6) + np.diag(sidejumpseven,k=7)
    H0 += np.diag(periodjump, k=29)
    
    H0 += H0.transpose().conjugate()
    return H0  


if __name__ == "__main__":
    
    length = 10


    t = -1
    U = 0. #Code is designed for U=0
    cc=0.142
    
    kxs = np.linspace(-np.pi,np.pi,500)

    energies = []
    '''np.set_printoptions(threshold=np.inf)
    print(Tb_Haiku_Infinite(U,t,np.pi/2))'''
    for kval in kxs:
        energies.append(eigvalsh(Tb_Haiku_Infinite(U,t,kval)))
    
    bands = np.sort(energies)
    halfgap = np.min(np.abs(bands))
    
    plt.figure()
    plt.title("Haiku-AGNR")
    plt.plot(kxs,bands[:,len(bands[0])//2-4:len(bands[0])//2+4], 'k')
    plt.show()