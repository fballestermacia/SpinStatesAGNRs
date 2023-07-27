import numpy as np
import matplotlib.pyplot as plt
from TB_GNR import GrapheGENEArmchair, lorentzian
from junction_XYZ import junction_xyz
from scipy.linalg import eigh, eigvalsh

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


def TB_From_XYZ(U,t,cc,matxyz): #cc in nanometers
    natoms = len(matxyz)
    
    Htb = np.diag(U/2*np.ones(natoms))
    
    for current_site in range(natoms):
        currentxyx = np.array(matxyz[current_site])
        for neighbour in range(current_site,natoms):
            dist = np.linalg.norm(currentxyx-np.array(matxyz[neighbour]))
            if np.isclose(dist,cc): 
                Htb[current_site,neighbour] = t
        printProgressBar(current_site+1,natoms, prefix="Consructing T-B Hamiltonian")
    
    Htb += np.transpose(Htb)
    
    return Htb
    
    
if __name__ == '__main__':
    t = -1
    Utb = 0
    cc = 0.142
    
    width1 = 25
    length1 = 20
    width2 = 13
    length2 = 5
    
    matxyz,xmat,ymat =  junction_xyz(width1,length1,width2,length2,cc, centered = True)
    Htbfrommat = TB_From_XYZ(Utb,t,cc,matxyz)
    
    
    eigvalsf, eigvect = eigh(Htbfrommat)
    
    eigvect = np.transpose(eigvect)
    
    g = 0.02
    res = 3000
    xvals = np.linspace(-3,3,res)
    dos = np.zeros(res)

    for eiv in eigvalsf:
        dos[:] += lorentzian(xvals[:],np.real(eiv),g)
    
    plt.figure()
    plt.plot(xvals, dos/np.max(dos), 'k')
    
    
    indices = [0,1,2,3]
    
    for indice in indices:
        plt.figure()
        plt.subplot(211)
        plt.scatter(xmat,ymat,25,c="r")  
        plt.title("w1 = {}, w2={}".format(width1,width2))
        p =len(xmat)//2-1-indice
        plt.scatter(xmat,ymat,s=np.square(np.abs(eigvect[p]))*100/np.max(np.square(np.abs(eigvect[p]))),c=np.square(np.abs(eigvect[p])))  
        plt.xlabel('x(nm)')
        plt.ylabel('y(nm)')
        plt.ylim(-0.1,cc*width1)
        
        plt.subplot(212)
        plt.scatter(xmat,ymat,25,c="r")  
        plt.title("w1 = {}, w2={}".format(width1,width2))
        p =len(xmat)//2+indice 
        plt.scatter(xmat,ymat,s=np.square(np.abs(eigvect[p]))*100/np.max(np.square(np.abs(eigvect[p]))),c=np.square(np.abs(eigvect[p])))  
        plt.xlabel('x(nm)')
        plt.ylabel('y(nm)')
        plt.ylim(-0.1,cc*width1)
    
    
    plt.show()