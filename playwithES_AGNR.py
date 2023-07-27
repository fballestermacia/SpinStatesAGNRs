import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, eigvalsh
from TB_GNR import TB_HamArmchair_finite, TB_HamArmchair_infinite, GrapheGENEArmchair


def edgestateDist_TB_AGNR(width, length, U,t):
    eigvals, eigvect = eigh(TB_HamArmchair_finite(width, length,U,t))
    eigvect = np.transpose(eigvect)
    eigvalsinf = eigvalsh(TB_HamArmchair_infinite(width,U,t,0)) #the gap is always smaller at k=0
    halfgap = np.min(np.abs(eigvalsinf))
    EdgestatesDists= []
    EdgestatesEnergies = []
    Nedgestates = 0
    if length ==1:
        EdgestatesDists.append(eigvect[width*length-1])
        EdgestatesEnergies.append(eigvals[width*length-1])
        EdgestatesDists.append(eigvect[width*length])
        EdgestatesEnergies.append(eigvals[width*length])
    for l,(energy, state) in enumerate(zip(eigvals, eigvect)):
        if np.abs(energy) <halfgap:
            Nedgestates += 1  
            EdgestatesDists.append(state)
            EdgestatesEnergies.append(energy)
    return Nedgestates, EdgestatesEnergies, EdgestatesDists




if __name__ == '__main__':
    width = 13
    length = 5

    scale= 150

    t = -1
    U = 0.
    cc=0.142

    Nedgestates, EdgestatesEnergies, EdgestatesDists = edgestateDist_TB_AGNR(width, length, U,t)

    print("There are {} edge states".format(Nedgestates))
    print(EdgestatesEnergies)
    matxyz,xmat,ymat =  GrapheGENEArmchair(width,length,0.142)


    p1s = [0,1]
    p2s = [-1,-2]

    for p1, p2 in zip(p1s,p2s):
        plt.figure()
        plt.subplot(211)
        plt.scatter(xmat,ymat,10,c="k",marker="x")  
        plt.title("Bonding")
        if np.max(EdgestatesDists[p1][::2*length]) < np.abs(np.min(EdgestatesDists[p1][::2*length])):
            colores = -np.real(EdgestatesDists[p1])
        else: colores = np.real(EdgestatesDists[p1])
        plt.scatter(xmat,ymat,s=np.square(np.abs(EdgestatesDists[p1]))*scale/np.max(np.square(np.abs(EdgestatesDists[p1]))),c=colores, cmap="bwr_r",vmin=-np.max(np.abs(colores)), vmax=np.max(np.abs(colores)))  
        plt.colorbar()
        #plt.xlabel('x(nm)')
        plt.ylabel('y(nm)')
        plt.ylim(-0.1,cc*width)

        plt.subplot(212)
        plt.scatter(xmat,ymat,10,c="k",marker="x")    
        plt.title("Antibonding")

        if np.max(EdgestatesDists[p2][::2*length]) < np.abs(np.min(EdgestatesDists[p2][::2*length])):
            colores = -np.real(EdgestatesDists[p2])
        else: colores = np.real(EdgestatesDists[p2])
        plt.scatter(xmat,ymat,s=np.square(np.abs(EdgestatesDists[p2]))*scale/np.max(np.square(np.abs(EdgestatesDists[p2]))),c=colores, cmap="bwr_r",vmin=-np.max(np.abs(colores)), vmax=np.max(np.abs(colores))) 
        plt.colorbar() 
        plt.xlabel('x(nm)')
        plt.ylabel('y(nm)')
        plt.ylim(-0.1,cc*width)


        plt.figure()

        plt.subplot(211)
        plt.scatter(xmat,ymat,10,c="k",marker="x")  
        plt.title("Left")

        if np.max(EdgestatesDists[p1][::2*length]) < np.abs(np.min(EdgestatesDists[p1][::2*length])):
            colores = (np.real(-EdgestatesDists[p1]) + np.real(EdgestatesDists[p2]))/2**0.5
            ss = np.square(np.abs(-EdgestatesDists[p1] + EdgestatesDists[p2]))*scale/np.max(np.square(np.abs(EdgestatesDists[p1] - EdgestatesDists[p2])))
        else: 
            colores = (np.real(EdgestatesDists[p1]) + np.real(EdgestatesDists[p2]))/2**0.5
            ss = np.square(np.abs(EdgestatesDists[p1] + EdgestatesDists[p2]))*scale/np.max(np.square(np.abs(EdgestatesDists[p1] + EdgestatesDists[p2])))
        
        plt.scatter(xmat,ymat,s=ss,c=colores, cmap="bwr_r",vmin=-np.max(np.abs(colores)), vmax=np.max(np.abs(colores)))  
        plt.colorbar()
        #plt.xlabel('x(nm)')
        plt.ylabel('y(nm)')
        plt.ylim(-0.1,cc*width)

        plt.subplot(212)
        plt.scatter(xmat,ymat,10,c="k",marker="x")    
        plt.title("Rigth")
        
        if np.max(EdgestatesDists[p1][::2*length]) > np.abs(np.min(EdgestatesDists[p1][::2*length])):
            colores = (np.real(EdgestatesDists[p1]) - np.real(EdgestatesDists[p2]))/2**0.5
            ss = np.square(np.abs(-EdgestatesDists[p1] + EdgestatesDists[p2]))*scale/np.max(np.square(np.abs(EdgestatesDists[p1] - EdgestatesDists[p2])))
        else: 
            colores = (np.real(EdgestatesDists[p1]) + np.real(EdgestatesDists[p2]))/2**0.5
            ss = np.square(np.abs(EdgestatesDists[p1] + EdgestatesDists[p2]))*scale/np.max(np.square(np.abs(EdgestatesDists[p1] + EdgestatesDists[p2])))
        plt.scatter(xmat,ymat,s=ss,c=colores, cmap="bwr_r",vmin=-np.max(np.abs(colores)), vmax=np.max(np.abs(colores))) 
        plt.colorbar() 
        plt.xlabel('x(nm)')
        plt.ylabel('y(nm)')
        plt.ylim(-0.1,cc*width)

    plt.show()