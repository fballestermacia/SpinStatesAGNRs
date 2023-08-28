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
    width1 =7
    length1 = 19
    
    width2 = 7
    length2 = 1
    
    t=-1
    U=1.2*np.abs(t)#1.2*np.abs(t)
    
    matxyzs,xmats,ymats =  junction_xyz(width1, length1, width2, length2, cc, centered = True, three=False)
    
    deletedX = []
    deletedY = []
    
    
    '''delposes = np.arange(length2*2)+width1*length1*2
    for delpos in delposes[::-1]:
        matxyzs = np.delete(matxyzs,delpos, axis = 0)
        xmats = np.delete(xmats,delpos)
        ymats = np.delete(ymats,delpos)'''
        
    
    delpos = 140
    matxyzs = np.delete(matxyzs,delpos, axis = 0)
    deletedX.append(xmats[delpos])
    deletedY.append(ymats[delpos])
    xmats = np.delete(xmats,delpos)
    ymats = np.delete(ymats,delpos)
    
    delpos = 130
    matxyzs = np.delete(matxyzs,delpos, axis = 0)
    deletedX.append(xmats[delpos])
    deletedY.append(ymats[delpos])
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
    
    '''nupinAVG[width1:width1//2*width1+1:2*width1] = 1
    ndowninAVG[-width2//2*width2:-width2:2*width2] = 1'''
    
    nupinAVG[-1] = 1
    #ndowninAVG[2*length1*17] = 1
    ndowninAVG[0] = 1
    
    
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
    
    fig, ax = plt.subplots()
    #ax.set_title("w1 = {}, w2={}. Energy = {} (units of $t$)".format(width1,width2, round(Energy,4)))
    ax.set_title("w = {}. Energy = {} (units of $t$)".format(width1, round(Energy,4)))
    #plt.scatter(xmats,ymats,10,c="k",marker="x")
    
    midpoint = np.max(xmats)/2
    window = 0.5
    #ax.set_xlim(midpoint*(1-window), midpoint*(1+window))
    
    color = nupout-ndownout
    size = nupout+ndownout
    
    ml = np.sum(color[:len(color)//2])
    mr = np.sum(color[len(color)//2:])
    print("m_l = ",ml, ' m_r = ', mr, ' m = ', ml+mr)
    
    checkmax = []
    for i,x in enumerate(xmats):
        if x > midpoint*(1-window) and x < midpoint*(1+window):
            checkmax.append(color[i])
    
    #print(np.min(size),np.max(size))
    #im = ax.scatter(xmats,ymats,s=size*100/np.max(size),c=color,alpha=1, vmin = np.min(checkmax), vmax = np.max(checkmax), cmap="bwr_r", edgecolors='black') 
    im = ax.scatter(xmats,ymats,s=size*100/np.max(size),c=color,alpha=1, vmin = -np.max(np.abs(checkmax)), vmax = np.max(np.abs(checkmax)), cmap="bwr_r", edgecolors='black') 
    #alpha=np.maximum(np.abs(color/np.max(np.abs(color)+0.000001)),0.3*np.ones(len(color)))
    
        
        
    cbar = fig.colorbar(im, ax=ax)
    
    plt.scatter(deletedX,deletedY, s=90, c='g', marker='x')
    
    plt.xlabel('x(nm)')
    plt.ylabel('y(nm)')
    
    plt.show()