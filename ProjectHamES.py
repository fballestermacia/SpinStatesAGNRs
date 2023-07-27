import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, eigvalsh
from TB_GNR import TB_HamArmchair_finite, GrapheGENEArmchair, lorentzian
from playwithES_AGNR import edgestateDist_TB_AGNR
import time
from datetime import timedelta, datetime
from Hubbard_GNR import SCF_Loop, HubbardHam_AGNR,fdstat
from Magnetcomparison import Ualpha, fsolve, ftosolveq
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

def fermifunc_effective(nel, resa, resb, ea, eb, KbT,interval=[-100,100], initialguess = 1, maxiter=5e3, precission = 1e-15):
    iter = 0
    Efant = 0
    Efpost = (np.max((ea+eb)/2)+np.min((ea+eb)/2))/2
    na = 0
    nb = 0
    liminf, limsup = interval[0], interval[1]
    while iter < maxiter and np.abs(Efant-Efpost) >precission:
        iter += 1
        na = np.sum(fdstat(ea[:],Efpost,KbT))
        nb = np.sum(fdstat(eb[:],Efpost,KbT))
        if na+nb < nel:
            liminf = Efpost
            dummy = Efpost
            Efpost = (Efpost+limsup)/2
            Efant = dummy
        elif na+nb > nel:
            limsup = Efpost
            dummy = Efpost
            Efpost = (Efpost+liminf)/2
            Efant = dummy
        elif na+nb == nel:
            Efant = Efpost
    else:
        maAVG = np.zeros(nel)
        mbAVG = np.zeros(nel)
        for i in range(nel):
            maAVG[i] = np.sum(np.square(np.abs(resa[:,i]))*fdstat(ea[:],Efpost,KbT))
            mbAVG[i] = np.sum(np.square(np.abs(resb[:,i]))*fdstat(eb[:],Efpost,KbT))


    return Efpost,maAVG,mbAVG, iter



def SCF_Loop_effective(Htb, nupinAVG,ndowninAVG, U, nel, alpha, Kbt, precission = 1e-6, maxiter=1e3):
    
    maant = np.zeros(nel)
    mbant = np.zeros(nel)
    
    maIn = np.copy(nupinAVG-ndowninAVG)
    mbIn = np.copy(nupinAVG-ndowninAVG)
    iter = 0   
    while (np.any(np.abs(maIn-maant) > precission) or np.any(np.abs(mbIn-mbant) > precission))and iter < maxiter:
        iter += 1
        
        
        Ha = np.matmul(np.diag(mbIn),U) + Htb 
        Hb = np.matmul(np.diag(maIn),U) + Htb
        
        psiaout = eigh(Ha) #REMEMBER TO TRANSPOSE DE EIGENVECTORS
        psibout = eigh(Hb)
         
        ea,resa = psiaout[0], np.transpose(psiaout[1])
        eb,resb = psibout[0], np.transpose(psibout[1])
        
        ef,maant,mbant,iterfermi = fermifunc_effective(nel, resa, resb, ea, eb, KbT)

        maIn = (1-alpha)*maIn + alpha*maant
        mbIn = (1-alpha)*mbIn + alpha*mbant
    #print("Iterations = ", iter, ', ', iterfermi)
    return maIn #(np.abs(2*(maIn-0.5))**0.5)


if __name__ == '__main__':
    
    save = True
    t=-1
    U = 1.2*np.abs(t)
    
    lengths = np.arange(40)+1
    
    cc = 0.142
    
    
    width = 13
    
    
    KbT = np.abs(t*1e-3)
    alpha = 0.5
    init_time = time.time()
    ms = []
    msright = []
    p = (width-1)//6
    ms = []
    Utheo = []
    Ueff = []
    for l,length in enumerate(lengths):
        nel = 2*width*length
        
        Htb = TB_HamArmchair_finite(width,length,0,t)
        
        psi = eigh(Htb)
        
        eup, dist = psi[0], np.transpose(psi[1])
        
        
        EdgeBState,EdgeABState =  dist[width*length-p:width*length],dist[width*length:width*length+p]
        LEdgeStates = []
        REdgeStates = []
        
        '''for BS, ABS in zip(EdgeBState,EdgeABState):
            if np.max(BS[::2*length]) < np.abs(np.min(BS[::2*length])):  
                LEdgeStates.append((BS-ABS)/2**0.5)
                REdgeStates.append((BS+ABS)/2**0.5)
            else:
                LEdgeStates.append((BS+ABS)/2**0.5)
                REdgeStates.append((BS-ABS)/2**0.5)'''
        
        states = []
        #for lstate,rstate in zip(LEdgeStates,REdgeStates):
        '''for lstate,rstate in zip(EdgeBState,EdgeABState):
            states.append(lstate)
            states.append(rstate)'''
            
        for lstate in EdgeBState:
            states.append(lstate)
        for lstate in EdgeABState:
            states.append(lstate)
            
            
        newHam = np.zeros((2*p,2*p))
        Ualphas =np.zeros((2*p,2*p))
        for i in range(2*p):
            for j in range(2*p):
                newHam[i,j] = np.matmul(np.transpose(states[i]),np.matmul(Htb,states[j]))
                Ualphas[i,j] = U/4*np.sum(np.abs(states[i])**2*np.abs(states[j])**2) 
                
        
        nel = 2*p
        
        qalphas = []
        alphas = np.arange((width+1)/2)+1
        kys = np.pi/(width+1)*alphas[-2:-((width-1)//3)//2-2:-1]

        dummylength = 100
        for ky in kys:
            qalphadummy,thetadummy = fsolve(ftosolveq,[0.8,-10], args=(ky,t,dummylength),xtol=1e-14)
            qalphas.append(qalphadummy)
        
        
        
        Ualphas2=[]
        for q in qalphas:
            Udum = Ualpha(q,width,length,1)
            Ualphas2.append(Udum)
            Ualphas2.append(Udum)
        
        
        
        
        Utheo.append(Ualphas2)
        Ueff.append(eigvalsh(Ualphas))
        #print(eigvalsh(Ualphas))
        
        Us = Ualphas#np.sum(Ualphas,axis=1)
        #print(Us)
        #print(Us)
        
        nupinAVG = np.zeros(nel)
        ndowninAVG = np.zeros(nel)
        
        nupinAVG[:p] = 1
        ndowninAVG[p:] = 1
        tot = np.sum(nupinAVG+ndowninAVG)
        #print(tot)
        nupinAVG = nupinAVG*nel/tot   
        ndowninAVG = ndowninAVG*nel/tot 
        #print(ndowninAVG)
        
        matxyz,xmat,ymat =  GrapheGENEArmchair(width,length,0.142)

        Vpotential = 0
        
        malpha = SCF_Loop_effective(newHam, nupinAVG,ndowninAVG, Us, nel, alpha,KbT, precission=1e-10)
        
        #print("malpha=",malpha)
        
        
        
        #print(nupout, ndownout, ef)
        mi = []
        for i in range(2*p):
            mi.append([np.abs(states[i])**2*malpha[i]])
          
        mL = 0 
        for mag in mi[p:]:
            mL += np.abs(np.sum(np.array(mag).reshape((width,2*length))[:,:length]))
        #print(mL)
        ms.append(2*np.sum(malpha[p:]))
        
        
        printProgressBar(l+1,len(lengths))
        #print(mL)
    
    print("Elapsed time: {}".format(timedelta(seconds=(time.time()-init_time))))
    
        
    pairs = np.array([lengths,ms]).transpose()
    if save:
        np.savetxt("dataMagnetizationProjected\width{}attemperature{}withDanielNotesat{}.dat".format(width,KbT,datetime.now().strftime('%d_%m_%Y-%H.%M.%S')),pairs,
                header="Hubbard Model edge magnetization with respect to length of ribbon. Armchair GNR of width = {} and temperature KbT = {}\n length (Unit Cells)       magnetic moment (Bohr Magneton units)".format(width,KbT)) 
    
    plt.figure()
    plt.plot(lengths,Utheo[:],'r')
    plt.plot(lengths,Ueff[:],'b')
    
    plt.figure()
    plt.title("w={}".format(width))
    plt.scatter(lengths,ms,c='k')
    plt.ylim(0-0.1,p*1.1)
    #plt.scatter(lengths,msright,c='k')
    plt.xlabel('Length')
    plt.ylabel('Edge magnetization')
    
    plt.show()