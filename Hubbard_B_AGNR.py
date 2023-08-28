import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, eigvalsh
from TB_GNR import TB_HamArmchair_finite, GrapheGENEArmchair, lorentzian
from junction_XYZ import junction_xyz
from TB_From_XYZ import TB_From_XYZ
from Hubbard_GNR import SCF_Loop, HubbardHam_AGNR, fdstat
from GNR_xyz import GrapheGENEHaiku
from TB_B_AGNR import GaussianInteraction
import time
from datetime import timedelta


if __name__ == '__main__':
    t = -1
    Utb = 0
    cc = 0.142
    U=1.2*np.abs(t)
    
    width = 7
    length = 31
    V = -np.abs(t)*1.42#1.42
    sigma = 0.15#0.15
    
    matxyz,xmat,ymat =  GrapheGENEArmchair(width,length,cc)
    
    borindexes = [width*length-1]#, width*length]
    borposx = np.zeros(len(borindexes))
    borposy = np.zeros(len(borindexes))
    
    onsiteEnergy = np.zeros(len(xmat))
    for i in range(len(borindexes)):

        borposx[i] = xmat[borindexes[i]][0]
        borposy[i] = ymat[borindexes[i]][0]
        distances = np.linalg.norm((matxyz-np.array([borposx[i],borposy[i]])), axis=1)
        onsiteEnergy += GaussianInteraction(V,distances,sigma)
    
    
        
    
    
    Htb = TB_From_XYZ(Utb,t,cc,matxyz) + np.diag(onsiteEnergy)
    
    Q =0#0.5
    Origin = (-10,(np.max(ymat)+np.min(ymat))/2)
    Vpotential = [Q,Origin] #0

    KbT = np.abs(t*1e-3) 
    alpha = 0.3
    
    nel = len(xmat)
    
    nupinAVG = np.zeros(nel)
    ndowninAVG = np.zeros(nel)
    
    nupinAVG[width:width//2*width+1:2*width] = 1
    ndowninAVG[-width//2*width:-width:2*width] = 1
    
    nupinAVG[width*length-1] = 1
    ndowninAVG[width*length] = 1
    
    
    tot = np.sum(nupinAVG+ndowninAVG)
    nupinAVG = (nupinAVG.flatten())*nel/tot   
    ndowninAVG = (ndowninAVG.flatten())*nel/tot 
    
    init_time = time.time()
    nupout, ndownout, ef, Htb2 = SCF_Loop(Htb, nupinAVG,ndowninAVG, U, KbT, nel, alpha,precission=1e-6, V= 0, xmat=np.array(xmat).flatten(), printea=True)
    Hup, Hdown = HubbardHam_AGNR(Htb2,nupout,ndownout,U)
    
    
    
    g = 0.05
    res = 3000
    dis = 4
    xvals = np.linspace(ef-dis,ef+dis,res)
    dosup = np.zeros(res)
    dosdown = np.zeros(res)

    bandup = eigvalsh(Hup)
    banddown = eigvalsh(Hdown)
    
    Energy = 0
    for eu,ed in zip(bandup,banddown):
        Energy += eu*fdstat(eu,ef,KbT) + ed*fdstat(ed,ef,KbT)
        #print(eu,ed,Energy)
    print("Fermi Energy = ", ef)
    print("Energy = ", Energy)      

    for eivup, eivdown in zip(bandup,banddown):
        dosup[:] += lorentzian(xvals[:],np.real(eivup),g)
        dosdown[:] -= lorentzian(xvals[:],np.real(eivdown),g)



    print("Elapsed time: {}".format(timedelta(seconds=(time.time()-init_time))))
    print("Up:{}, Down:{}, Total:{}, Number of electrons:{}".format(np.sum(nupout), np.sum(ndownout), np.sum(nupout + ndownout), nel))

    plt.figure()
    plt.title("w1 = {}. Energy = {} (units of $t$)".format(width, round(Energy,4)))
    plt.plot(xvals, dosup, 'b', label="$\\uparrow$")
    plt.plot(xvals, dosdown, 'r', label="$\\downarrow$")
    plt.vlines(ef, np.min(dosdown),np.max(dosup))
    plt.legend()
    
    plt.figure()
    plt.title("w1 = {}. Energy = {} (units of $t$)".format(width, round(Energy,4)))
    plt.scatter(xmat,ymat,10,c="k",marker="x")

    
    color = nupout-ndownout
    size = nupout+ndownout
    
    #print(np.min(size),np.max(size))
    plt.scatter(xmat,ymat,s=size*100/np.max(size),c=color,alpha=1, cmap="bwr_r", vmin=-np.max(np.abs(color)), vmax=np.max(np.abs(color)),edgecolors='black') 
    
    #alpha=np.maximum(np.abs(color/np.max(np.abs(color)+0.000001)),0.3*np.ones(len(color)))
    
    plt.colorbar()
    plt.xlabel('x(nm)')
    plt.ylabel('y(nm)')
    plt.scatter(borposx, borposy, marker='x', c='k')
    
    
    
    plt.show()