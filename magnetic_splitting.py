import numpy as np
import matplotlib.pyplot as plt
from Hubbard_GNR import SCF_Loop, HubbardHam_AGNR,fdstat
from scipy.linalg import eigh, eigvalsh
from TB_GNR import TB_HamArmchair_finite, GrapheGENEArmchair, lorentzian
import time
from datetime import timedelta
from playwithES_AGNR import edgestateDist_TB_AGNR
import warnings

warnings.filterwarnings('ignore')

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



def EnergyandDensity(Htb,nup,ndown,efs,U,KbT, width, length):
    Hup, Hdown = HubbardHam_AGNR(Htb,nup,ndown,U)

    bandup = eigvalsh(Hup)
    banddown = eigvalsh(Hdown)
    Energy = 0
    for eu,ed in zip(bandup,banddown):
        Energy += eu*fdstat(eu,efs,KbT) + ed*fdstat(ed,efs,KbT)
    
    dleft = np.sum(nup.reshape(width,2*length)[:,:length] + ndown.reshape(width,2*length)[:,:length])
    mleft = np.sum(nup.reshape(width,2*length)[:,:length] - ndown.reshape(width,2*length)[:,:length])
    
    dright = np.sum(nup.reshape(width,2*length)[:,length:] + ndown.reshape(width,2*length)[:,length:])
    mright = np.sum(nup.reshape(width,2*length)[:,length:] - ndown.reshape(width,2*length)[:,length:])
    
    return Energy, dleft, dright, mleft, mright



if __name__ == '''__main__''':
    cc = 0.142
    width = 19
    length = 40
    t=-1
    U=1.2*np.abs(t)
    nel = 2*width*length
    
    matxyz,xmat,ymat =  GrapheGENEArmchair(width,length,0.142)
    Origin = (-10,(np.max(ymat)+np.min(ymat))/2)
    
    

    KbT = np.abs(t*1e-3) 
    alpha = 0.3
    
    resq = 50
    Qs = np.linspace(0,5,resq)
    
    EP = np.empty(resq)
    EAP = np.empty(resq)
    MP = np.empty(resq)
    MAP = np.empty(resq)
    
    DLP = np.empty(resq)
    DRP = np.empty(resq)
    DLAP = np.empty(resq)
    DRAP = np.empty(resq)
    
    Htb = TB_HamArmchair_finite(width,length,0,t)
    Nedgestates = int((width-1)/6)
    es, dist = eigh(Htb)
    ESdist = np.transpose(dist)[width*length-Nedgestates:width*length]
    
    
    printProgressBar(0,resq)
    for qindex, Q in enumerate(Qs):
        Vpotential = [Q,Origin]
        
        alpha = 0.3
        
        
        ##########
        #Parallel
        ##########
        nupinp = np.zeros(nel).reshape((width,2*length))
        ndowninp = np.zeros(nel).reshape((width,2*length))
        for state in ESdist:
            state = state.reshape(width,2*length)
            nupinp[:,:length] += np.abs(state)[:,:length]
            ndowninp[:,length:] += np.abs(state)[:,length:]
        tot = np.sum(nupinp+ndowninp)
        nupinp = (nupinp.flatten())/tot
        ndowninp = (ndowninp.flatten())/tot
            
        
          
                
        nupoutp, ndownoutp, efp, Htb2 = SCF_Loop(Htb, nupinp,ndowninp, U, KbT, nel, alpha,maxiter=1e2, precission=1e-4, V= Vpotential, xmat=np.array(xmat).flatten(), printea=False,symmetricpotential=True)
        printProgressBar(2*qindex+1,2*resq)
        
        ep,dleftp,drightp,mlp,mrp = EnergyandDensity(Htb2,nupoutp,ndownoutp,efp,U,KbT, width, length)
       
       
        ##########
        #Antiparallel
        ##########
        nupinap  = np.zeros(nel).reshape((width,2*length))
        ndowninap  = np.zeros(nel).reshape((width,2*length))
        '''for i,state in enumerate(ESdist):
            state = state.reshape(width,2*length)
            if not i%2:
                nupinap [:,length:] += np.abs(state)[:,length:]
                ndowninap [:,:length] += np.abs(state)[:,:length]
            else:
                nupinap [:,:length] += np.abs(state)[:,:length]
                ndowninap [:,length:] += np.abs(state)[:,length:]'''
        nupinap[1,0] = 1
        nupinap[width//2-1,0] = 1
        nupinap[-2,-1] = 1
        
        ndowninap[1,-1] = 1
        ndowninap[width//2-1,-1] = 1
        ndowninap[-2,-1] = 1
        
        tot = np.sum(nupinap+ndowninap)
        nupinap = (nupinap.flatten())/tot 
        ndowninap  = (ndowninap.flatten())/tot       
       
        nupoutap, ndownoutap, efap, Htb2 = SCF_Loop(Htb, nupinap ,ndowninap , U, KbT, nel, alpha,maxiter=1e2, precission=1e-4, V= Vpotential, xmat=np.array(xmat).flatten(), printea=False,symmetricpotential=True)
        
        eap,dleftap,drightap, mlap, mrap = EnergyandDensity(Htb2,nupoutap,ndownoutap,efap,U,KbT,width,length)
        
        if np.abs(mlap) > 1.5 or np.abs(mlap)<0.5:
            fuckthishit = True
            while fuckthishit:
                alpha = 0.4*np.random.random(1)+0.1
                print("Not correctly converged, trying again...")
                nupinap  = np.zeros(nel).reshape((width,2*length))
                ndowninap  = np.zeros(nel).reshape((width,2*length))
                for i,state in enumerate(ESdist):
                    state = state.reshape(width,2*length)
                    if not i%2:
                        nupinap [:,length:] += np.abs(state)[:,length:]
                        ndowninap [:,:length] += np.abs(state)[:,:length]
                    else:
                        nupinap [:,:length] += np.abs(state)[:,:length]
                        ndowninap [:,length:] += np.abs(state)[:,length:]
                tot = np.sum(nupinap+ndowninap)
                nupinap = (nupinap.flatten())/tot 
                ndowninap  = (ndowninap.flatten())/tot       
            
                nupoutap, ndownoutap, efap, Htb2 = SCF_Loop(Htb, nupinap ,ndowninap , U, KbT, nel, alpha,maxiter=1e2, precission=1e-4, V= Vpotential, xmat=np.array(xmat).flatten(), printea=False,symmetricpotential=True)
                
                eap,dleftap,drightap, mlap, mrap = EnergyandDensity(Htb2,nupoutap,ndownoutap,efap,U,KbT,width,length)
                if np.abs(mlap) > 1.5 or np.abs(mlap)<0.5: 
                    pass
                else: fuckthishit=False
            
        
        EP[qindex] = ep
        EAP[qindex] = eap
        MP[qindex] = np.abs(mlp)
        MAP[qindex] = np.abs(mlap)
        DLP[qindex] = dleftp
        DRP[qindex] = drightp
        DLAP[qindex] = dleftap
        DRAP[qindex] = drightap
        
        
        printProgressBar(2*qindex+2,2*resq)
        print("\n Q = ",Q,' m = ',mlap)
    
    mids = (EP+EAP)/2
    
    
    
    plt.figure()
    plt.title("Energies. $r_0=${}".format(Origin))
    plt.plot(Qs,EP,'b')
    plt.plot(Qs,EAP,'r')
    
    
    plt.figure()
    plt.title("Energy splitting. $r_0=${}".format(Origin))
    plt.plot(Qs,EP-mids,'b')
    plt.plot(Qs,EAP-mids,'r')
    
    
    plt.figure()
    plt.title("Densities")
    plt.plot(Qs,DLP,'b',label='DLP')
    plt.plot(Qs,DLAP,'b--', label='DLAP')
    plt.plot(Qs,DRP,'r',label='DRP')
    plt.plot(Qs,DRAP,'r--', label='DRAP')
    
    plt.figure()
    plt.title("Magnetic moments")
    plt.scatter(Qs,MP,c='b',label='MP')
    plt.scatter(Qs,MAP,c='r', label='MAP')
    
    plt.show()
        
        
        
        