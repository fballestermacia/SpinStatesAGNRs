import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, eigvalsh
from TB_GNR import TB_HamArmchair_finite, GrapheGENEArmchair, lorentzian
import time
from datetime import timedelta
from playwithES_AGNR import edgestateDist_TB_AGNR

def fdstat(e,ef,KbT):
    return 1/(np.exp((e-ef)/KbT)+1)


def HubbardHam_AGNR(Htb, nupAVG, ndownAVG, U, Bfield=0):
    Hup = U*np.diag(ndownAVG) + Htb + Bfield*np.diag(np.ones(len(nupAVG)))
    Hdown = U*np.diag(nupAVG) + Htb - Bfield*np.diag(np.ones(len(ndownAVG)))
    
    return Hup, Hdown

def fermifunc(nel, spinup, spindown, eup, edown, KbT,interval=[-100,100], initialguess = 0.6, maxiter=5e3, precission = 1e-15, printea = True, halffilled=False):
    iter = 0
    Efant = 0
    Efpost = initialguess
    nup = 0
    ndown = 0
    liminf, limsup = interval[0], interval[1]
    while iter < maxiter and np.abs(Efant-Efpost) >precission:
        iter += 1
        if printea: print("\rFermi Function iteration: {}".format(iter), end='')
        nup = np.sum(fdstat(eup[:],Efpost,KbT))
        ndown = np.sum(fdstat(edown[:],Efpost,KbT))
        if nup+ndown < nel:
            
            liminf = Efpost
            dummy = Efpost
            Efpost = (Efpost+limsup)/2
            Efant = dummy
        elif nup+ndown > nel:
            
            limsup = Efpost
            dummy = Efpost
            Efpost = (Efpost+liminf)/2
            Efant = dummy
        elif nup+ndown == nel:
            Efant = Efpost
        
    if halffilled:
        nupAVG = np.zeros(nel*2)
        ndownAVG = np.zeros(nel*2)
    else: 
        nupAVG = np.zeros(nel)
        ndownAVG = np.zeros(nel)
    for i in range(nel):
        nupAVG[i] = np.sum(np.square(np.abs(spinup[:,i]))*fdstat(eup[:],Efpost,KbT))
        ndownAVG[i] = np.sum(np.square(np.abs(spindown[:,i]))*fdstat(edown[:],Efpost,KbT))
    if printea: print("\n \n")
    
    return Efpost, nupAVG, ndownAVG, iter

'''
def fermifunc(nel, spinup, spindown, eup, edown, KbT, initialguess = 1, maxiter=5e3, precission = 1e-5, printea = True, halffilled=False):
    iter = 0
    Ef2ant = 0
    Efant = 0
    Efpost = initialguess
    if halffilled:
        nupAVG = np.zeros(nel*2)
        ndownAVG = np.zeros(nel*2)
    else: 
        nupAVG = np.zeros(nel)
        ndownAVG = np.zeros(nel)
    while iter < maxiter and np.abs(Efant-Efpost) >precission:
        iter += 1 
        if printea: print("\rFermi Function iteration: {}".format(iter), end='')
        #print(iter,np.abs(Efant-Efpost), Efpost)
        for i in range(nel):
            nupAVG[i] = np.sum(np.square(np.abs(spinup[:,i]))*fdstat(eup[:],Efpost,KbT))
            ndownAVG[i] = np.sum(np.square(np.abs(spindown[:,i]))*fdstat(edown[:],Efpost,KbT))
        if np.sum(nupAVG+ndownAVG) <= nel:
            if Efant > Efpost:
                dummy = Efpost
                Efpost = (Efpost+Efant)/2
                Ef2ant = Efant
                Efant = dummy
            elif Ef2ant > Efpost:
                dummy = Efpost
                Efpost = (Efpost+Ef2ant)/2
                Ef2ant = Efant
                Efant = dummy
            else: 
                Ef2ant = Efant
                Efant = Efpost
                Efpost = Efpost*2
        elif np.sum(nupAVG+ndownAVG) > nel:
            if Efant < Efpost:
                dummy = Efpost
                Efpost = (Efpost+Efant)/2
                Ef2ant = Efant
                Efant = dummy
            elif Ef2ant < Efpost:
                dummy = Efpost
                Efpost = (Efpost+Ef2ant)/2
                Ef2ant = Efant
                Efant = dummy
            else: 
                Ef2ant = Efant
                Efant = Efpost
                Efpost = (0+Efpost)/2
    for i in range(nel):
        
        nupAVG[i] = np.sum(np.square(np.abs(spinup[:,i]))*fdstat(eup[:],Efpost,KbT))
        ndownAVG[i] = np.sum(np.square(np.abs(spindown[:,i]))*fdstat(edown[:],Efpost,KbT))
    #we should normalize the densities
    #ndummy = np.sum(nupAVG+ndownAVG)
    #nupAVG = nel*nupAVG/ndummy
    #ndownAVG = nel*ndownAVG/ndummy
    if printea: print("\n")
    return Efpost, nupAVG, ndownAVG, iter
'''


def SCF_Loop(Htb, nupAVG,ndownAVG, U, KbT, nel, alpha, precission = 1e-6, maxiter=5e2, V=0., xmat=np.empty(1), ymat=np.empty(1), printea = True, Bfield = 0, halffilled=False, iguessEf=1, Umatrix=False, symmetricpotential=False):
    
    if V != 0.: 
        if type(V) is float: #this is to simulate (i guess) a potentiel gradient along the x direction
            if symmetricpotential: 
                Htb2 = Htb+np.diag(V*xmat) + np.diag(V*xmat[::-1])
            else: Htb2 = Htb+np.diag(V*xmat)
        else: # V = [Q, Origin] (origin = distance from leftmost edge)
            charge = V[0]
            xorigin, yorigin = V[1]
            Xmat2 = xmat - xorigin 
            Ymat2 = ymat - yorigin
            Htb2 = Htb+np.diag(-charge*np.power((Xmat2**2+Ymat2**2)**0.5,-np.ones(len(Xmat2))))
            
            if symmetricpotential: 
                Xmat3 = xmat - (np.max(xmat) - xorigin)
                Htb2 = Htb2+np.diag(-charge*np.power((Xmat3**2+Ymat2**2)**0.5,-np.ones(len(Xmat2))))
    else: Htb2 = np.copy(Htb)
            
    
    nupantAVG = np.zeros(nel)
    ndownantAVG = np.zeros(nel)
    
    iter = 0
    if type(U) is float:
        ef = U/2
    else: ef = iguessEf#np.sum(U)/(nel*2)
    
    while (np.any(np.abs(nupAVG-nupantAVG) > precission) or np.any(np.abs(ndownAVG-ndownantAVG) > precission)) and iter < maxiter:
        if printea: print(np.sum(np.abs(nupAVG-nupantAVG)>precission)or np.any(np.abs(ndownAVG-ndownantAVG) > precission))
        iter += 1
        if printea: print("\rSCF loop iteration = {}".format(iter), end='\n')
        #if Umatrix:
        #    Hup, Hdown = HubbardHam_AGNR_UMatrix(Htb2,nupAVG,ndownAVG,U)
        #else:
        Hup, Hdown = HubbardHam_AGNR(Htb2,nupAVG,ndownAVG,U, Bfield=Bfield)
        psiupout = eigh(Hup) #REMEMBER TO TRANSPOSE DE EIGENVECTORS
        psidownout = eigh(Hdown)
        
        
        eup,spinup = psiupout[0], np.transpose(psiupout[1])
        edown,spindown = psidownout[0], np.transpose(psidownout[1])
        #print(fermifunc(nel,spinup,spindown,eup,edown,KbT))
        if halffilled:
            ef,nupresAVG,ndownresAVG,iterfermifunc = fermifunc(nel//2,spinup,spindown,eup,edown,KbT, printea=printea,halffilled=halffilled)
        else: ef,nupresAVG,ndownresAVG,iterfermifunc = fermifunc(nel,spinup,spindown,eup,edown,KbT, printea=printea,initialguess=ef)
        nupantAVG = np.copy(nupAVG)
        ndownantAVG = np.copy(ndownAVG)
        #print(nupresAVG-ndownresAVG)
        nupAVG = (1-alpha)*nupAVG + alpha*nupresAVG
        ndownAVG = (1-alpha)*ndownAVG + alpha*ndownresAVG
    if printea: print("\n")
    return nupAVG, ndownAVG, ef, Htb2
        
        
if __name__ == '__main__':
    cc = 0.142
    width = 19
    length = 20
    t=-1
    U=1.2*np.abs(t)#1.2*np.abs(t)
    nel = 2*width*length
    
    matxyz,xmat,ymat =  GrapheGENEArmchair(width,length,0.142)
    
    B=0#-1e-6
    
    Q =50#0.5
    Origin = (-10,(np.max(ymat)+np.min(ymat))/2)
    Vpotential = [Q,Origin] #0

    KbT = np.abs(t*1e-3) 
    alpha = 0.3


    '''nupinAVG = 0.5*np.ones(nel)
    ndowninAVG = 1-nupinAVG'''

    #Nedgestates, EdgestatesEnergies, EdgestatesDists = edgestateDist_TB_AGNR(width, length, 0,t)
    Htb = TB_HamArmchair_finite(width,length,0,t)
    Nedgestates = int((width-1)/6)
    es, dist = eigh(Htb)
    ESdist = np.transpose(dist)[width*length-Nedgestates:width*length]

    nupinAVG = 0.5*np.ones(nel).reshape((width,2*length))
    ndowninAVG = 0.5*np.ones(nel).reshape((width,2*length))
    
    '''for i,state in enumerate(ESdist):
        state = state.reshape(width,2*length)
        if not i%2:
            nupinAVG[:,length:] += np.abs(state)[:,length:]
            ndowninAVG[:,:length] += np.abs(state)[:,:length]
            nupinAVG[:,:length] -= np.abs(state)[:,:length]
            ndowninAVG[:,length:] -= np.abs(state)[:,length:]
            print("State ", i, " down left.")
        else:
            nupinAVG[:,:length] += np.abs(state)[:,:length]
            ndowninAVG[:,length:] += np.abs(state)[:,length:]
            nupinAVG[:,length:] -= np.abs(state)[:,length:]
            ndowninAVG[:,:length] -= np.abs(state)[:,:length]
            print("State ", i, " up left.")'''
    
    '''nupinAVG = 0.55*np.ones(nel).reshape((width,2*length))
    ndowninAVG = 0.45*np.ones(nel).reshape((width,2*length))'''
    
    '''state = ESdist[3].flatten()

    nupinAVG = nupinAVG.flatten()
    ndowninAVG = ndowninAVG.flatten()

    for i in range(len(state)):
        if state[i] > 0:
            nupinAVG[i] = np.real(state[i])
        else:
            ndowninAVG[i] = -np.real(state[i])
'''
    '''left = (EdgestatesDists[0]+EdgestatesDists[-1]).flatten()/2**0.5
    right = (EdgestatesDists[0]-EdgestatesDists[-1]).flatten()/2**0.5

    nupinAVG = np.abs(left)
    ndowninAVG = np.abs(right)'''
    
    
    nupinAVG = np.zeros(nel).reshape((width,2*length))
    ndowninAVG = np.zeros(nel).reshape((width,2*length))
    
    
    
    nupinAVG[1::8,0] = 1
    nupinAVG[3::8,0] = 1
    ndowninAVG[5::8,0] = 1
    
    ndowninAVG[1::8,-1] = 1
    ndowninAVG[3::8,-1] = 1
    nupinAVG[5::8,-1] = 1
    
    ''' nupinAVG[1::2,0] = 1
    ndowninAVG[1::2,-1] = 1'''
    
    '''nupinAVG[1,0] = 1
    nupinAVG[width//2-1,0] = 1
    nupinAVG[-2,-1] = 1
    
    ndowninAVG[1,-1] = 1
    ndowninAVG[width//2-1,-1] = 1
    ndowninAVG[-2,-1] = 1'''
    
    tot = np.sum(nupinAVG+ndowninAVG)
    
    nupinAVG = (nupinAVG.flatten())*nel/tot   
    ndowninAVG = (ndowninAVG.flatten())*nel/tot 


    Htb = TB_HamArmchair_finite(width,length,0,t)
    init_time = time.time()
    nupout, ndownout, ef, Htb2 = SCF_Loop(Htb, nupinAVG,ndowninAVG, U, KbT, nel, alpha, precission=1e-4, V= Vpotential, xmat=np.array(xmat).flatten(), printea=True, symmetricpotential=True, Bfield=B)
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
    Energy -= U*np.sum(nupout*ndownout)    
    
    print("Fermi Energy = ", ef)
    print("Energy = ", Energy)      

    for eivup, eivdown in zip(bandup,banddown):
        dosup[:] += lorentzian(xvals[:],np.real(eivup),g)
        dosdown[:] -= lorentzian(xvals[:],np.real(eivdown),g)



    print("Elapsed time: {}".format(timedelta(seconds=(time.time()-init_time))))
    print("Up:{}, Down:{}, Total:{}, Number of electrons:{}".format(np.sum(nupout), np.sum(ndownout), np.sum(nupout + ndownout), nel))

    plt.figure()
    plt.title("w = {}, l={}. Energy = {} (units of $t$)".format(width,length, round(Energy,4)))
    plt.plot(xvals, dosup, 'b', label="$\\uparrow$")
    plt.plot(xvals, dosdown, 'r', label="$\\downarrow$")
    plt.vlines(ef, np.min(dosdown),np.max(dosup))
    plt.legend()



    matxyz,xmat,ymat =  GrapheGENEArmchair(width,length,0.142)
    
    plt.figure()
    plt.title("w = {}, l={}. Energy = {} (units of $t$)".format(width,length,round(Energy,4)))
    plt.scatter(xmat,ymat,10,c="k",marker="x")

    color = nupout-ndownout
    
    mleft = nupout.reshape((width,2*length))[:,:length]- ndownout.reshape((width,2*length))[:,:length]
    print("m_l = ",np.sum(mleft))

    size = nupout+ndownout
    #print(np.min(size),np.max(size))
    plt.scatter(xmat,ymat,s=size*100/np.max(size),c=color,alpha=1, cmap="bwr_r", vmin=-np.max(np.abs(color)), vmax=np.max(np.abs(color)),edgecolors='black') 
    
    #alpha=np.maximum(np.abs(color/np.max(np.abs(color)+0.000001)),0.3*np.ones(len(color)))
    
    plt.colorbar()
    plt.xlabel('x(nm)')
    plt.ylabel('y(nm)')

    psiupout = eigh(Hup) #REMEMBER TO TRANSPOSE DE EIGENVECTORS
    psidownout = eigh(Hdown)
    eup,spinup = psiupout[0], np.transpose(psiupout[1])
    edown,spindown = psidownout[0], np.transpose(psidownout[1])
    
    ps = 0#(width-1)//6
    nmedio = width*length

    for p in range(ps):
        plt.figure()
        
        plt.subplot(211)
        plt.title("up"+str(p))
        plt.scatter(xmat,ymat,10,c="k",marker="x")  
        
        colores = np.real(spinup[nmedio-1-p])
        plt.scatter(xmat,ymat,s=np.square(np.abs(spinup[nmedio-1-p]))*100/np.max(np.square(np.abs(spinup[nmedio-1-p]))),c=colores, cmap="bwr_r",vmin=-np.max(np.abs(colores)), vmax=np.max(np.abs(colores)))  
        plt.colorbar()
        plt.xlabel('x(nm)')
        plt.ylabel('y(nm)')
        plt.ylim(-0.1,cc*width)
        
        plt.subplot(212)
        plt.scatter(xmat,ymat,10,c="k",marker="x")  
        
        colores = np.real(spinup[nmedio-1-p])
        plt.scatter(xmat,ymat,s=np.square(np.abs(spinup[nmedio+p]))*100/np.max(np.square(np.abs(spinup[nmedio+p]))),c=colores, cmap="bwr_r",vmin=-np.max(np.abs(colores)), vmax=np.max(np.abs(colores)))  
        plt.colorbar()
        plt.xlabel('x(nm)')
        plt.ylabel('y(nm)')
        plt.ylim(-0.1,cc*width)
        
        plt.figure()
        
        plt.subplot(211)
        plt.title("down"+str(p))
        plt.scatter(xmat,ymat,10,c="k",marker="x")  
        
        colores = np.real(spindown[nmedio-1-p])
        plt.scatter(xmat,ymat,s=np.square(np.abs(spindown[nmedio-1-p]))*100/np.max(np.square(np.abs(spindown[nmedio-1-p]))),c=colores, cmap="bwr_r",vmin=-np.max(np.abs(colores)), vmax=np.max(np.abs(colores)))  
        plt.colorbar()
        plt.xlabel('x(nm)')
        plt.ylabel('y(nm)')
        plt.ylim(-0.1,cc*width)
        
        plt.subplot(212)
        plt.scatter(xmat,ymat,10,c="k",marker="x")  
        
        colores = np.real(spindown[nmedio-1-p])
        plt.scatter(xmat,ymat,s=np.square(np.abs(spindown[nmedio+p]))*100/np.max(np.square(np.abs(spindown[nmedio+p]))),c=colores, cmap="bwr_r",vmin=-np.max(np.abs(colores)), vmax=np.max(np.abs(colores)))  
        plt.colorbar()
        plt.xlabel('x(nm)')
        plt.ylabel('y(nm)')
        plt.ylim(-0.1,cc*width)
    
    plt.show()