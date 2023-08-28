import numpy as np
import matplotlib.pyplot as plt
from TB_GNR import GrapheGENEArmchair, lorentzian, TB_HamArmchair_infinite
from junction_XYZ import junction_xyz
from GNR_xyz import GrapheGENEHaiku
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


def TB_From_XYZ(U,t,cc,matxyz, printea=True): #cc in nanometers
    natoms = len(matxyz)
    
    Htb = np.diag(U/2*np.ones(natoms))
    
    for current_site in range(natoms):
        currentxyx = np.array(matxyz[current_site])
        for neighbour in range(current_site,natoms):
            dist = np.linalg.norm(currentxyx-np.array(matxyz[neighbour]))
            if np.isclose(dist,cc): 
                Htb[current_site,neighbour] = t
        if printea:printProgressBar(current_site+1,natoms, prefix="Consructing T-B Hamiltonian")
    
    Htb += np.transpose(Htb)
    
    return Htb
    
    
if __name__ == '__main__':
    t = -1
    Utb = 0
    cc = 0.142
    
    width1 = 7
    length1 = 19
    width2 = 7
    length2 = 1
    
    centered = True
    
    matxyz,xmat,ymat =  junction_xyz(width1,length1,width2,length2,cc, centered = centered, three=True) #GrapheGENEHaiku(length1,cc)
    
    delpos = 129
    matxyz = np.delete(matxyz,delpos, axis = 0)
    xmat = np.delete(xmat,delpos)
    ymat = np.delete(ymat,delpos)
    
    delpos = 140
    matxyz = np.delete(matxyz,delpos, axis = 0)
    xmat = np.delete(xmat,delpos)
    ymat = np.delete(ymat,delpos)
    
    Htbfrommat = TB_From_XYZ(Utb,t,cc,matxyz)
    
    print(len(xmat), ' electrons')
    
    energies = eigvalsh(TB_HamArmchair_infinite(np.max([width1,width2]),Utb,t,0))

    bands = np.sort(energies)
    halfgap = np.min(np.abs(bands))
    
    NES = 0
    
    eigvalsf, eigvect = eigh(Htbfrommat)
    
    eigvect = np.transpose(eigvect)
    
    for en in eigvalsf:
        if np.abs(en) < halfgap: NES +=1
    
    print(NES, " Edge States")
    
    g = 0.02
    res = 3000
    xvals = np.linspace(-3,3,res)
    dos = np.zeros(res)

    for eiv in eigvalsf:
        dos[:] += lorentzian(xvals[:],np.real(eiv),g)
    
    plt.figure()
    plt.plot(xvals, dos/np.max(dos), 'k')
    plt.vlines(halfgap, 0,1)
    plt.vlines(-halfgap, 0,1)
    
    
    indices = np.arange(NES//2)
    
    
    for indice in indices:
        if NES%2:
            plt.figure()
            plt.subplot(311)
            plt.scatter(xmat,ymat,10,c="k", marker='x')  
            #plt.title("w1 = {}, w2={}".format(width1,width2))
            plt.title("w = {}".format(width1))
            p =len(xmat)//2-1-indice
            print(eigvalsf[p])
            plt.scatter(xmat,ymat,s=np.square(np.abs(eigvect[p]))*100/np.max(np.square(np.abs(eigvect[p]))),c=eigvect[p], cmap='bwr_r')  
            #plt.xlabel('x(nm)')
            plt.ylabel('y(nm)')
            plt.ylim(-0.1+np.min(ymat),0.1+np.max(ymat))
            plt.colorbar()
            
            plt.subplot(312)
            plt.scatter(xmat,ymat,10,c="k", marker='x')   
            #plt.title("w1 = {}, w2={}".format(width1,width2))
            p =len(xmat)//2+1+indice
            print(eigvalsf[p])
            plt.scatter(xmat,ymat,s=np.square(np.abs(eigvect[p]))*100/np.max(np.square(np.abs(eigvect[p]))),c=eigvect[p], cmap='bwr_r')  
            #plt.xlabel('x(nm)')
            plt.ylabel('y(nm)')
            plt.ylim(-0.1+np.min(ymat),0.1+np.max(ymat))
            plt.colorbar()
            
            plt.subplot(313)
            plt.scatter(xmat,ymat,10,c="k", marker='x')  
            #plt.title("w1 = {}, w2={}".format(width1,width2))
            p =len(xmat)//2+indice 
            print(eigvalsf[p])
            plt.scatter(xmat,ymat,s=np.square(np.abs(eigvect[p]))*100/np.max(np.square(np.abs(eigvect[p]))),c=eigvect[p], cmap='bwr_r')  
            plt.xlabel('x(nm)')
            plt.ylabel('y(nm)')
            plt.ylim(-0.1+np.min(ymat),0.1+np.max(ymat))
            plt.colorbar()
        else:
            plt.figure()
            plt.subplot(211)
            plt.scatter(xmat,ymat,10,c="k", marker='x')  
            plt.title("w1 = {}, w2={}".format(width1,width2))
            p =len(xmat)//2-1-indice
            print(eigvalsf[p])
            plt.scatter(xmat,ymat,s=np.square(np.abs(eigvect[p]))*100/np.max(np.square(np.abs(eigvect[p]))),c=eigvect[p], cmap='bwr_r')  
            #plt.xlabel('x(nm)')
            plt.ylabel('y(nm)')
            plt.ylim(-0.1+np.min(ymat),0.1+np.max(ymat))
            plt.colorbar()
            
            midpoint = np.max(xmat)/2
            window = 0.5
            #plt.xlim(midpoint*(1-window), midpoint*(1+window))
            
            plt.subplot(212)
            plt.scatter(xmat,ymat,10,c="k", marker='x')   
            #plt.title("w1 = {}, w2={}".format(width1,width2))
            p =len(xmat)//2+indice
            print(eigvalsf[p])
            plt.scatter(xmat,ymat,s=np.square(np.abs(eigvect[p]))*100/np.max(np.square(np.abs(eigvect[p]))),c=eigvect[p], cmap='bwr_r')  
            plt.xlabel('x(nm)')
            plt.ylabel('y(nm)')
            plt.ylim(-0.1+np.min(ymat),0.1+np.max(ymat))
            plt.colorbar()
            #plt.xlim(midpoint*(1-window), midpoint*(1+window))
            
    
    plt.show()