import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.linalg import eigh, eigvalsh
from TB_GNR import TB_HamArmchair_finite, GrapheGENEArmchair
from Hubbard_GNR import SCF_Loop, HubbardHam_AGNR, lorentzian


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


if __name__ == "__main__":
    width = 7
    length = 20
    
    t=-1
    
    resol = 50
    Us = np.abs(t)*np.linspace(0.1,3.5,resol)
    
    nel = 2*width*length
    
    matxyz,xmat,ymat =  GrapheGENEArmchair(width,length,0.142)
    
    Q = 0
    Origin = (-5,(np.max(ymat)+np.min(ymat))/2)
    Vpotential = [Q,Origin] 

    KbT = np.abs(t*1e-3) 
    alpha = 0.3
    
    Htb = TB_HamArmchair_finite(width,length,0,t)
    Nedgestates = int((width-1)/6)
    es, dist = eigh(Htb)
    ESdist = np.transpose(dist)[width*length-Nedgestates:width*length]

    nupinAVG = np.zeros(nel).reshape((width,2*length))
    ndowninAVG = np.zeros(nel).reshape((width,2*length))
    
    for i,state in enumerate(ESdist):
        state = state.reshape(width,2*length)
        if False:#not i%2:
            nupinAVG[:,length:] += np.abs(state)[:,length:]
            ndowninAVG[:,:length] += np.abs(state)[:,:length]
            print("State ", i, " down left.")
        else:
            nupinAVG[:,:length] += np.abs(state)[:,:length]
            ndowninAVG[:,length:] += np.abs(state)[:,length:]
            print("State ", i, " up left.")
    
    tot = np.sum(nupinAVG+ndowninAVG)
    nupinAVG = (nupinAVG.flatten())/tot 
    ndowninAVG = (ndowninAVG.flatten())/tot
    
    g = 0.05
    res = 3000
    dis = 1.2
    
    xvals = np.linspace(-dis,dis,res)
    
    DOS = []
    gaps = []
    for i,U in enumerate(Us):
        nupout, ndownout, ef, Htb2 = SCF_Loop(Htb, nupinAVG,ndowninAVG, U, KbT, nel, alpha, precission=1e-6, V= Vpotential, xmat=np.array(xmat).flatten(), printea=False)
        Hup, Hdown = HubbardHam_AGNR(Htb2,nupout,ndownout,U)
        
        
        
        dosup = np.zeros(res)
        dosdown = np.zeros(res)

        bandup = eigvalsh(Hup)
        banddown = eigvalsh(Hdown)
        midup = (bandup[len(bandup)//2]+bandup[len(bandup)//2-1])/2
        for eivup, eivdown in zip(bandup,banddown):
            dosup[:] += lorentzian(xvals[:],np.real(eivup)-midup,g)
        
        DOS.append(dosup)
        gaps.append(np.min(np.abs(bandup-midup)))
        printProgressBar(i+1,resol)
    
    EVALS,UVALS = np.meshgrid(xvals,Us)

    plt.figure()
    
    plt.scatter(Us,np.array(gaps),c='b')
    
    
    fig = plt.figure()
    ax = plt.axes(projection ='3d')

    ax.plot_surface(EVALS,UVALS,np.array(DOS), cmap=cm.summer)
    
    ax.set_xlabel("$E$ (centered at 0)")
    ax.set_ylabel("$U$")
    ax.set_zlabel("DOS")

    plt.show()