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
    width = 25
    length = 20
    t=-1
    U=1.2*np.abs(t)
    nel = 2*width*length
    
    matxyz,xmat,ymat =  GrapheGENEArmchair(width,length,0.142)
    Origin = (-1,(np.max(ymat)+np.min(ymat))/2)
    
    

    KbT = np.abs(t*1e-3) 
    alpha = 0.3
    
    resq = 10
    Bs = np.linspace(0,1e-4,resq)
    
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
    for qindex, B in enumerate(Bs):
        #Vpotential = [Q,Origin]
        
        alpha = 0.3
        
        
        ##########
        #Parallel
        ##########
        nupinp = np.zeros(nel).reshape((width,2*length))
        ndowninp = np.zeros(nel).reshape((width,2*length))
        '''for state in ESdist:
            state = state.reshape(width,2*length)
            nupinp[:,:length] += np.abs(state)[:,:length]
            ndowninp[:,length:] += np.abs(state)[:,length:]
        
        
            '''
        nupinp[1::8,0] = 1
        nupinp[3::8,0] = 1
        nupinp[5::8,0] = 1
        nupinp[7::8,0] = 1
        ndowninp[1::8,-1] = 1
        ndowninp[3::8,-1] = 1
        ndowninp[5::8,-1] = 1
        ndowninp[7::8,-1] = 1
        
        tot = np.sum(nupinp+ndowninp)
        nupinp = (nupinp.flatten())/tot*nel
        ndowninp = (ndowninp.flatten())/tot*nel
                
        nupoutp, ndownoutp, efp, Htb2 = SCF_Loop(Htb, nupinp,ndowninp, U, KbT, nel, alpha,maxiter=1e2, precission=1e-4, xmat=np.array(xmat).flatten(), printea=False,Bfield=-B)
        printProgressBar(2*qindex+1,2*resq)
        
        ep,dleftp,drightp,mlp,mrp = EnergyandDensity(Htb2,nupoutp,ndownoutp,efp,U,KbT, width, length)
       
        print("\n mp = ", mlp)
       
        ##########
        #Antiparallel
        ##########
        nupinap = 0.5*np.zeros(nel).reshape((width,2*length))
        ndowninap = 0.5*np.zeros(nel).reshape((width,2*length))
        
        '''for i,state in enumerate(ESdist):
            state = state.reshape(width,2*length)
            if not i%2:
                nupinap[:,length:] += np.abs(state)[:,length:]**2
                ndowninap[:,:length] += np.abs(state)[:,:length]**2
                nupinap[:,:length] -= np.abs(state)[:,:length]**2
                ndowninap[:,length:] -= np.abs(state)[:,length:]**2
                
            else:
                nupinap[:,:length] += np.abs(state)[:,:length]**2
                ndowninap[:,length:] += np.abs(state)[:,length:]**2'''
                
        
        '''nupinap[1,0] = 1
        nupinap[width//2-1,0] = 1
        nupinap[-2,-1] = 1
        
        ndowninap[1,-1] = 1
        ndowninap[width//2-1,-1] = 1
        ndowninap[-2,-1] = 1'''
        
        nupinap[1::8,0] = 1
        nupinap[3::8,0] = 1
        ndowninap[5::8,0] = 1
        ndowninap[7::8,0] = 1
        ndowninap[1::8,-1] = 1
        ndowninap[3::8,-1] = 1
        nupinap[5::8,-1] = 1
        nupinap[7::8,-1] = 1
        
        tot = np.sum(nupinap+ndowninap)
        nupinap = (nupinap.flatten())/tot*nel
        ndowninap  = (ndowninap.flatten())/tot*nel       
       
        nupoutap, ndownoutap, efap, Htb2 = SCF_Loop(Htb, nupinap ,ndowninap , U, KbT, nel, alpha,maxiter=1e2, precission=1e-4, xmat=np.array(xmat).flatten(), printea=False,Bfield=B)
        
        eap,dleftap,drightap, mlap, mrap = EnergyandDensity(Htb2,nupoutap,ndownoutap,efap,U,KbT,width,length)
        
        '''if np.abs(mlap)>1.2 or np.abs(mlap)<0.8:
            fuckthishit = True
            counter = 0
            while fuckthishit:
                if counter > 5: break
                alpha = np.random.random(1)
                print("Not correctly converged, trying again...")
                print(mlap)
                nupinap = 0.5*np.ones(nel).reshape((width,2*length))
                ndowninap = 0.5*np.ones(nel).reshape((width,2*length))
                
                for i,state in enumerate(ESdist):
                    state = state.reshape(width,2*length)
                    if not i%2:
                        nupinap[:,length:] += np.abs(state)[:,length:]**2
                        ndowninap[:,:length] += np.abs(state)[:,:length]**2
                        nupinap[:,:length] -= np.abs(state)[:,:length]**2
                        ndowninap[:,length:] -= np.abs(state)[:,length:]**2
                        
                    else:
                        nupinap[:,:length] += np.abs(state)[:,:length]**2
                        ndowninap[:,length:] += np.abs(state)[:,length:]**2
                        
                        
                tot = np.sum(nupinap+ndowninap)
                nupinap = (nupinap.flatten())/tot*nel 
                ndowninap  = (ndowninap.flatten())/tot*nel       
            
                nupoutap, ndownoutap, efap, Htb2 = SCF_Loop(Htb, nupinap ,ndowninap , U, KbT, nel, alpha,maxiter=1e2, precission=1e-4, xmat=np.array(xmat).flatten(), printea=False,Bfield=B)
        
                eap,dleftap,drightap, mlap, mrap = EnergyandDensity(Htb2,nupoutap,ndownoutap,efap,U,KbT,width,length)
                if np.abs(mlap)>1.2 or np.abs(mlap)<0.8:
                    counter +=1
                    pass
                else: fuckthishit=False'''
            
        
        EP[qindex] = ep
        EAP[qindex] = eap
        MP[qindex] = np.abs(mlp)
        MAP[qindex] = np.abs(mlap)
        DLP[qindex] = dleftp
        DRP[qindex] = drightp
        DLAP[qindex] = dleftap
        DRAP[qindex] = drightap
        
        
        printProgressBar(2*qindex+2,2*resq)
        print("\n B= ",B,' m = ',mlap)
    
    mids = (EP+EAP)/2
    
    
    
    plt.figure()
    plt.title("Energies. $r_0=${}".format(Origin))
    plt.plot(Bs,EP,'b')
    plt.plot(Bs,EAP,'r')
    
    
    plt.figure()
    plt.title("Energy splitting. $r_0=${}".format(Origin))
    plt.plot(Bs,EP-mids,'b')
    plt.plot(Bs,EAP-mids,'r')
    
    
    plt.figure()
    plt.title("Densities")
    plt.plot(Bs,DLP,'b',label='DLP')
    plt.plot(Bs,DLAP,'b--', label='DLAP')
    plt.plot(Bs,DRP,'r',label='DRP')
    plt.plot(Bs,DRAP,'r--', label='DRAP')
    
    plt.figure()
    plt.title("Magnetic moments")
    plt.scatter(Bs,MP,c='b',label='MP')
    plt.scatter(Bs,MAP,c='r', label='MAP')
    
    plt.show()
        
        
        
        