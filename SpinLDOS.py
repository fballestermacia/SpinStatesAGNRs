import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, eigvalsh
from TB_GNR import TB_HamArmchair_finite, GrapheGENEArmchair
from Hubbard_GNR import SCF_Loop, HubbardHam_AGNR,fdstat
import time
from datetime import timedelta


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


def edgespingi(E,POS, eigenergiesup, eigstatesup, eigenergiesdown, eigstatesdown, sigma,length):
    sumup = 0
    sumdown = 0
    for ie in range(len(eigenergiesup)):
        for cup,cdown in zip(eigstatesup,eigstatesdown):#CHANGE THIS
            sumup += np.abs(eigstatesup.flatten()[2*length*(1+2*POS)])**2*np.exp(-0.5*((E-eigenergiesup[ie])/sigma)**2)
            sumdown += np.abs(eigstatesdown.flatten()[2*length*(1+2*POS)])**2*np.exp(-0.5*((E-eigenergiesdown[ie])/sigma)**2)
    
    return sumup, sumdown



if __name__ == "__main__":
    cc = 0.142
    width = 25
    length = 15
    t=-1
    U=1.2*np.abs(t)
    nel = 2*width*length
    
    KbT = np.abs(t*1e-3) 
    alpha = 0.8
    
    Htb = TB_HamArmchair_finite(width,length,0,t)
    
    nupinAVG = np.zeros(nel).reshape((width,2*length))
    ndowninAVG = np.zeros(nel).reshape((width,2*length))
    
    nupinAVG[1,0] = 1
    nupinAVG[-2,-1] = 1
    ndowninAVG[1,-1] = 1
    ndowninAVG[-2,0] = 1
    
    tot = np.sum(nupinAVG+ndowninAVG)   
    nupinAVG = (nupinAVG.flatten())/tot  
    ndowninAVG = (ndowninAVG.flatten())/tot
    
    matxyz,xmat,ymat =  GrapheGENEArmchair(width,length,0.142)
    init_time = time.time()
    nupout, ndownout, ef = SCF_Loop(Htb, nupinAVG,ndowninAVG, U, KbT, nel, alpha, precission=1e-3, V= 0, xmat=np.array(xmat).flatten())
    Hup, Hdown = HubbardHam_AGNR(Htb,nupout,ndownout,U)
    
    
    psiup = eigh(Hup)
    
    eigenergiesUP = psiup[0]
    eigstatesUP = np.transpose(psiup[1])
    
    psidown = eigh(Hdown)
    
    eigenergiesDOWN = psidown[0]
    eigstatesDOWN = np.transpose(psidown[1])
    
    res =50
    es = np.linspace(ef-0.2,ef+0.2,res)
    pos = np.arange((width-1)//2)
    
    POS, E = np.meshgrid(pos,es)
    
    
    sigma=1e-2
    
    
    LDOSup = np.empty((res,(width-1)//2))
    LDOSdown = np.empty((res,(width-1)//2))
    for i,ee in enumerate(es):
        for j in range(len(pos)):
            LDOSup[i,j], LDOSdown[i,j] = edgespingi(ee,pos[j],eigenergiesUP,nupout,eigenergiesDOWN,ndownout,sigma,length)
            printProgressBar(i,res)
    
    
    print("Elapsed time: {}".format(timedelta(seconds=(time.time()-init_time))))
    LDOSup /= np.max(np.abs(LDOSup))
    LDOSdown /= np.max(np.abs(LDOSdown))

    

    
    plt.figure()
    plt.plot(pos+1,np.transpose(np.sum(LDOSup,axis=0))/np.max(np.transpose(np.sum(LDOSup,axis=0))),'b.-', label="$\\uparrow$")
    plt.plot(pos+1,np.transpose(np.sum(LDOSdown,axis=0))/np.max(np.transpose(np.sum(LDOSdown,axis=0))),'r.-',label="$\\downarrow$")
    #plt.plot(POS+1,np.sum(LDOSdown,axis=1))
    #plt.contourf(POS+1,E,LDOSup-LDOSdown,13,cmap="bwr_r",vmin=-1,vmax=1)
    #plt.contourf(POS+1,E,LDOSdown,13,cmap="bwr",vmin=-1,vmax=1)
    #plt.colorbar()
    #plt.hlines(ef,1,np.max(POS+1))
    plt.legend()
    
    '''fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    
    surf1 = ax.plot_surface(POS+1, E, LDOSup, color='b',linewidth=0)
    surf2 = ax.plot_surface(POS+1, E, LDOSdown, color='r',linewidth=0)
    '''
    plt.xlabel("Position along the edge of the ribbon")
    plt.ylabel("Projected local density of states (a.u.)")
    plt.title("Spin-LDOS (a.u.). w = {}, l = {}".format(width, length))
    plt.show()