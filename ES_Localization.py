import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, eigvalsh
from TB_GNR import TB_HamArmchair_finite, TB_HamArmchair_infinite, GrapheGENEArmchair
from scipy.optimize import curve_fit, fsolve
from ESEnergies_Extended import ftosolveq
import warnings


if __name__=='__main__':
    
    t = -1
    width = 13
    length = 20
    nes =int((width-1)/6)
    
    qalphas = []
    alphas = np.arange((width+1)/2)+1
    #print(alphas[-2:-np.max(nes)//2-2:-1])
    kys = np.pi/(width+1)*alphas[-2:-nes-2:-1]
    #print(kys)
    
    for ky in kys:
        qalphadummy,thetadummy = fsolve(ftosolveq,[0.5,-10], args=(ky,t,152),xtol=1e-8)
        #print(qalphadummy,thetadummy)
        qalphas.append(qalphadummy)
    
    Htb = TB_HamArmchair_finite(width,length,0,t)
    
    en, dist = eigh(Htb)
    ESen = en[width*length-nes:width*length]
    ESdist = np.transpose(dist)[width*length-nes:width*length]
    EdgestatesDensity = []
    for l,edist in enumerate(ESdist):
        state = edist.reshape(width,2*length)
        denslength = np.zeros(4*length)
        for i in range(2*length):
            for j in range(2):
                denslength[2*i+(i+(1-j))%2] = np.linalg.norm(state[j::2,i])**2
                #print(len(state[j::2,i]))
                #print(np.square(np.abs(state[(1-j)::,i]))) 
        EdgestatesDensity.append((denslength))
    
    
        
    xlengths = np.linspace(0,2*length,4*length)
    xlengths2 = xlengths
    plt.figure()
    plt.title("Density of electrons along the length of the ribbon")
    #plt.hlines(0,xlengths[0],xlengths[-1],r)
    listacol = ['b','k','r','g','c']
    for l,esd in enumerate(EdgestatesDensity):
        plt.plot(xlengths,esd, color=listacol[l])
        plt.plot(xlengths,esd,'o', color=listacol[l])
        plt.plot(xlengths2,np.max(esd)*(np.exp(-2*qalphas[-1-l]*xlengths2)+np.exp(-2*qalphas[-1-l]*(2*length-xlengths2))),'--', color=listacol[l])
    plt.xlabel("Atom position")

    
    plt.ylabel("Density of electrons (a.u.)")
    plt.show()
