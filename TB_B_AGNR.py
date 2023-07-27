import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, eigvalsh
from TB_GNR import TB_HamArmchair_finite, GrapheGENEArmchair, lorentzian
from junction_XYZ import junction_xyz
from TB_From_XYZ import TB_From_XYZ
from Hubbard_GNR import SCF_Loop, HubbardHam_AGNR, fdstat
from GNR_xyz import GrapheGENEHaiku
from LDOS import gi, printProgressBar
import time
from datetime import timedelta

def GaussianInteraction(V, d, sigma):
    return V*np.exp(-d**2/(2*sigma**2))





if __name__ == '__main__':
    t = -1
    Utb = 0
    cc = 0.142
    
    width = 7
    length = 21
    V = 1.42*np.abs(t)
    sigma = 0.15
    
    matxyz,xmat,ymat =  GrapheGENEArmchair(width,length,cc)
    
    borindexes = [width*length-1, width*length]#, width*length-2]
    borposx = np.zeros(len(borindexes))
    borposy = np.zeros(len(borindexes))
    
    onsiteEnergy = np.zeros(len(xmat))
    for i in range(len(borindexes)):

        borposx[i] = xmat[borindexes[i]][0]
        borposy[i] = ymat[borindexes[i]][0]
        distances = np.linalg.norm((matxyz-np.array([borposx[i],borposy[i]])), axis=1)
        onsiteEnergy += GaussianInteraction(V,distances,sigma)
    
    
        
    
    
    Htbfrommat = TB_From_XYZ(Utb,t,cc,matxyz) + np.diag(onsiteEnergy)
    
    
    eigvalsf, eigvect = eigh(Htbfrommat)
    
    eigvect = np.transpose(eigvect)
    
    g = 0.02
    res = 3000
    xvals = np.linspace(-3,3,res)
    dos = np.zeros(res)

    for eiv in eigvalsf:
        dos[:] += lorentzian(xvals[:],np.real(eiv),g)
    
    plt.figure()
    plt.plot(xvals, dos/np.max(dos), 'k')
    
    
    indices = [0,1,2,3]
    
    for indice in indices:
        plt.figure()
        plt.subplot(211)
        plt.scatter(xmat,ymat,25,c="r")  
        plt.title("w1 = {}".format(width))
        p =len(xmat)//2-1-indice
        plt.scatter(xmat,ymat,s=np.square(np.abs(eigvect[p]))*100/np.max(np.square(np.abs(eigvect[p]))),c=np.square(np.abs(eigvect[p])))  
        plt.xlabel('x(nm)')
        plt.ylabel('y(nm)')
        plt.ylim(-0.1,cc*width)
        
        plt.scatter(borposx, borposy, marker='x', c='k')
        
        plt.subplot(212)
        plt.scatter(xmat,ymat,25,c="r")  
        plt.title("w1 = {}".format(width))
        p =len(xmat)//2+indice 
        plt.scatter(xmat,ymat,s=np.square(np.abs(eigvect[p]))*100/np.max(np.square(np.abs(eigvect[p]))),c=np.square(np.abs(eigvect[p])))  
        plt.xlabel('x(nm)')
        plt.ylabel('y(nm)')
        plt.ylim(-0.1,cc*width)
        plt.scatter(borposx, borposy, marker='x', c='k')
    
    sigma=2e-2
    
    
    
    res = 1000
    es = np.linspace(-3.5,3.5,res)
    pos = np.arange(4*length)
    
    POS, E = np.meshgrid(pos,es)
    
    LDOS = np.empty((res,4*length))
    for i,ee in enumerate(es):
        for j in range(2*length):
            if xmat[j]>xmat[j+2*length]:
                LDOS[i,2*j+1] = gi(ee,pos[j],eigvalsf,eigvect,sigma, length)
                LDOS[i,2*j] = gi(ee,pos[j+2*length],eigvalsf,eigvect,sigma, length)
            else:
                LDOS[i,2*j] = gi(ee,pos[j],eigvalsf,eigvect,sigma, length)
                LDOS[i,2*j+1] = gi(ee,pos[j+2*length],eigvalsf,eigvect,sigma, length)
            printProgressBar(i+1,res, prefix="Calculating LDOS")
    
    LDOS /= np.max(LDOS)
    
    plt.figure()
    plt.contourf(POS+1,E,LDOS,cmap="inferno")
    plt.colorbar()
    plt.xlabel("Position along the length of the ribbon (number of columns)")
    plt.ylabel("Energy (units of $t$)")
    plt.title("LDOS (a.u.). w = {}, l = {}".format(width, length))
    
    
    plt.show()