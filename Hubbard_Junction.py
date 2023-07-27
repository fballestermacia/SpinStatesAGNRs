import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, eigvalsh
from TB_GNR import TB_HamArmchair_finite, GrapheGENEArmchair, lorentzian
from junction_XYZ import junction_xyz
from TB_From_XYZ import TB_From_XYZ
from Hubbard_GNR import SCF_Loop, HubbardHam_AGNR, fdstat
from GNR_xyz import GrapheGENEHaiku
import time
from datetime import timedelta


if __name__ == '__main__':
    cc = 0.142
    width1 = 7
    length1 = 5
    
    width2 = 7
    length2 = 5
    
    t=-1
    U=1.2*np.abs(t)#1.2*np.abs(t)
    
    matxyzs,xmats,ymats =  junction_xyz(width1, length1, width2, length2, cc, centered = False)
    
    delpos = 38
    matxyzs = np.delete(matxyzs,delpos, axis = 0)
    xmats = np.delete(xmats,delpos)
    ymats = np.delete(ymats,delpos)
    
    delpos = 100
    matxyzs = np.delete(matxyzs,delpos, axis = 0)
    xmats = np.delete(xmats,delpos)
    ymats = np.delete(ymats,delpos)
    #matxyz,xmat,ymat =  GrapheGENEHaiku(length1,cc)
    
    Htb = TB_From_XYZ(0,t,cc,matxyzs)
    
    Q =0#0.5
    Origin = (-10,(np.max(ymats)+np.min(ymats))/2)
    Vpotential = [Q,Origin] #0

    KbT = np.abs(t*1e-3) 
    alpha = 0.3
    
    nel = len(xmats)
    
    nupinAVG = 0.5*np.zeros(nel)
    ndowninAVG = 0.5*np.zeros(nel)
    
    nupinAVG[width1:width1//2*width1+1:2*width1] = 1
    ndowninAVG[-width2//2*width2:-width2:2*width2] = 1
    
    
    
    tot = np.sum(nupinAVG+ndowninAVG)
    nupinAVG = (nupinAVG.flatten())*nel/tot   
    ndowninAVG = (ndowninAVG.flatten())*nel/tot 
    
    init_time = time.time()
    nupout, ndownout, ef, Htb2 = SCF_Loop(Htb, nupinAVG,ndowninAVG, U, KbT, nel, alpha,maxiter = 100,precission=1e-6, V= 0, xmat=np.array(xmats).flatten(), printea=True)
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
    plt.title("w1 = {}, w2={}. Energy = {} (units of $t$)".format(width1,width2, round(Energy,4)))
    plt.plot(xvals, dosup, 'b', label="$\\uparrow$")
    plt.plot(xvals, dosdown, 'r', label="$\\downarrow$")
    plt.vlines(ef, np.min(dosdown),np.max(dosup))
    plt.legend()
    
    plt.figure()
    plt.title("w1 = {}, w2={}. Energy = {} (units of $t$)".format(width1,width2, round(Energy,4)))
    plt.scatter(xmats,ymats,10,c="k",marker="x")

    
    color = nupout-ndownout
    size = nupout+ndownout
    
    #print(np.min(size),np.max(size))
    plt.scatter(xmats,ymats,s=size*100/np.max(size),c=color,alpha=1, cmap="bwr_r", vmin=-np.max(np.abs(color)), vmax=np.max(np.abs(color)),edgecolors='black') 
    
    #alpha=np.maximum(np.abs(color/np.max(np.abs(color)+0.000001)),0.3*np.ones(len(color)))
    
    plt.colorbar()
    plt.xlabel('x(nm)')
    plt.ylabel('y(nm)')
    
    plt.show()