import numpy as np
import matplotlib.pyplot as plt
from TB_GNR import GrapheGENEArmchair, lorentzian, TB_HamArmchair_infinite, TB_HamArmchair_finite
from junction_XYZ import junction_xyz
from GNR_xyz import GrapheGENEHaiku
from scipy.linalg import eigh, eigvalsh
from TB_From_XYZ import TB_From_XYZ

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
    t = -1
    Utb = 0
    cc = 0.142
    
    
    width1 = 9
    length1 = 30
    width2 = 7
    
    nlengths = 46
    lengths2 = np.arange(nlengths)+5
    
    centered = True
    
    Gapsjunction = np.empty(nlengths)
    Gapspristine = np.empty(nlengths)
    printProgressBar(0,nlengths)
    for i, length2 in enumerate(lengths2):
    
        matxyz,xmat,ymat =  junction_xyz(width1,length1,width2,length2,cc, centered = centered, three=True) 
        Htbfrommat = TB_From_XYZ(Utb,t,cc,matxyz, printea=False)
   
        eigvals = eigvalsh(Htbfrommat)

        p1 =len(xmat)//2-2
        p2 =len(xmat)//2+1 
        Gapsjunction[i] = np.abs(eigvals[p1]-eigvals[p2])
        
        Htbprist = TB_HamArmchair_finite(width2,length2,Utb,t)
        
        eigvalsp = eigvalsh(Htbprist)

        pp1 =width2*length2-1
        pp2 =width2*length2
        Gapspristine[i] = np.abs(eigvalsp[pp1]-eigvalsp[pp2])
        printProgressBar(i+1,nlengths)
        
    plt.figure()
    
    plt.plot(lengths2,Gapspristine,'r.-', label='Pristine Edge States')
    plt.plot(lengths2,Gapsjunction,'b.-',label='Junction States')
    
    
    plt.legend()
    
    plt.xlabel("Length (unit cells)")
    plt.ylabel("$\\delta$ in units of $t$")
    plt.yscale('log')
    
    plt.show()
    