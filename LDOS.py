import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, eigvalsh
from TB_GNR import TB_HamArmchair_finite, GrapheGENEArmchair

def gi(E,POS,eigenergies, eigstates, sigma, length):
    sum = 0
    for energy, state in zip(eigenergies,eigstates):
        for c in state.flatten()[POS::4*length]:
            sum += np.abs(c)**2*np.exp(-0.5*((E-energy)/sigma)**2)
    
    return sum

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
    length = 5
    
    t=-1
    U = 0
    
    Htb = TB_HamArmchair_finite(width,length,0,t)
    
    psi = eigh(Htb)
    
    eigenergies = psi[0]
    eigstates = np.transpose(psi[1])
    
    res = 1000
    es = np.linspace(-3.5,3.5,res)
    pos = np.arange(4*length)
    
    POS, E = np.meshgrid(pos,es)
    
    
    matxyz,xmat,ymat =  GrapheGENEArmchair(width,length,0.142)
    
    sigma=2e-2
    
    LDOS = np.empty((res,4*length))
    
    
    for i,ee in enumerate(es):
        for j in range(2*length):
            if xmat[j]>xmat[j+2*length]:
                LDOS[i,2*j+1] = gi(ee,pos[j],eigenergies,eigstates,sigma, length)
                LDOS[i,2*j] = gi(ee,pos[j+2*length],eigenergies,eigstates,sigma, length)
            else:
                LDOS[i,2*j] = gi(ee,pos[j],eigenergies,eigstates,sigma, length)
                LDOS[i,2*j+1] = gi(ee,pos[j+2*length],eigenergies,eigstates,sigma, length)
            printProgressBar(i,res)
    
    LDOS /= np.max(LDOS)
    
    plt.figure()
    plt.contourf(POS+1,E,LDOS,cmap="inferno")
    plt.colorbar()
    plt.xlabel("Position along the length of the ribbon (number of columns)")
    plt.ylabel("Energy (units of $t$)")
    plt.title("LDOS (a.u.). w = {}, l = {}".format(width, length))
    plt.show()
