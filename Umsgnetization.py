import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, eigvalsh
from TB_GNR import TB_HamArmchair_finite, GrapheGENEArmchair, lorentzian
from playwithES_AGNR import edgestateDist_TB_AGNR
import time
from datetime import timedelta, datetime
from Hubbard_GNR import SCF_Loop

# Print iterations progress
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


if __name__ == '__main__':
    
    
    t=-1
    Us = np.abs(t)*np.linspace(0,1.5,15)#1.2*np.abs(t)
    
    length = 40
    
    cc = 0.142
    width = 19
    nel = 2*width*length
    
    KbT = np.abs(t*1e-3)
    alpha = 0.2
    init_time = time.time()
    ms = []
    msright = []
    
    
    Htb = TB_HamArmchair_finite(width,length,0,t)
    Nedgestates = int((width-1)/6)
    es, dist = eigh(Htb)
    ESdist = np.transpose(dist)[width*length-Nedgestates:width*length]

    nupinAVG = 0.5*np.ones(nel).reshape((width,2*length))
    ndowninAVG = 0.5*np.ones(nel).reshape((width,2*length))
    
    for i,state in enumerate(ESdist):
        state = state.reshape(width,2*length)
        if not i%2:
            nupinAVG[:,length:] += np.abs(state)[:,length:]**2
            ndowninAVG[:,:length] += np.abs(state)[:,:length]**2
            nupinAVG[:,:length] -= np.abs(state)[:,:length]**2
            ndowninAVG[:,length:] -= np.abs(state)[:,length:]**2
            print("State ", i, " down left.")
        else:
            nupinAVG[:,:length] += np.abs(state)[:,:length]**2
            ndowninAVG[:,length:] += np.abs(state)[:,length:]**2
            nupinAVG[:,length:] -= np.abs(state)[:,length:]**2
            ndowninAVG[:,:length] -= np.abs(state)[:,:length]**2
            print("State ", i, " up left.")
    
    tot = np.sum(nupinAVG+ndowninAVG)
    #print(tot)
    nupinAVG = (nupinAVG.flatten())/tot#*nel/tot   
    ndowninAVG = (ndowninAVG.flatten())/tot#*nel/tot 
    
    matxyz,xmat,ymat =  GrapheGENEArmchair(width,length,0.142)

    Q =0#0.5
    Origin = (-10,(np.max(ymat)+np.min(ymat))/2)
    Vpotential = [Q,Origin] #0

    Htb = TB_HamArmchair_finite(width,length,0,t)
    
    for j,U in enumerate(Us):
        nupout, ndownout, ef, Htb2 = SCF_Loop(Htb, nupinAVG,ndowninAVG, U, KbT, nel, alpha, precission=1e-6, V= Vpotential, xmat=np.array(xmat).flatten(), printea=False)
        nleft = nupout.reshape((width,2*length))[:,:length]- ndownout.reshape((width,2*length))[:,:length]
        ms.append(np.sum(nleft)) 
        #print("Up:{}, Down:{}, Total:{}, Number of electrons:{}".format(np.sum(nupout), np.sum(ndownout), np.sum(nupout + ndownout), nel))
        #print(np.sum(nupout),np.sum(ndownout))
        printProgressBar(j+1,len(Us))
    
    save = False
    
    pairs = np.array([Us,ms]).transpose()
    if save: np.savetxt("dataUMagnetization\width{}length{}andTemperature{}at{}.dat".format(width,length,KbT,datetime.now().strftime('%d_%m_%Y-%H.%M.%S')),pairs,
               header="Hubbard Model edge magnetization with respect to Coulomb repulsion. Armchair GNR of width = {}, length = {} and temperature KbT = {}\n U/t        magnetic moment (Bohr Magneton units)".format(width,length,KbT)) 
    
    
    print("Elapsed time: {}".format(timedelta(seconds=(time.time()-init_time))))
    plt.figure()
    plt.title("w={},l={}".format(width, length))
    plt.scatter(Us,ms,c='k')
    plt.vlines(2*np.abs(t),0,np.max(ms))
    #plt.scatter(lengths,msright,c='k')
    plt.xlabel('U')
    plt.ylabel('Edge magnetization')
    
    plt.show()