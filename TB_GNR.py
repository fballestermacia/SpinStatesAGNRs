import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, eigvalsh



def lorentzian(x,x0,g): #this is just for plotting
    return 1/np.pi*(0.5*g/((x-x0)**2+(0.5*g)**2))


def GrapheGENEZigzag(width,length,cc):
    Natomsx = 2*length
    Natomsy = width
    matxyz = []
    xmat = []
    ymat = []
    for xi in np.arange(Natomsx):
        for yj in np.arange(Natomsy):
            i = xi+1
            j = yj+1
            parimparJ = divmod(j,2)
            parimparJf=parimparJ[1]
            parimparI = divmod(i+1,2)
            parimparIf=parimparI[1]
            x = (i-1)*np.sqrt(3)/2*cc
            y = (3*(j-1)+parimparJf+((-1)**(j))*parimparIf)*cc/2
            matxyz.append([x,y])
            xmat.append([x])
            ymat.append([y])
    return matxyz,xmat,ymat


def GrapheGENEArmchair(width,length,cc):
    Natomsy = 2*length
    Natomsx = width
    matxyz = []
    xmat = []
    ymat = []
    for xi in np.arange(Natomsx):
        for yj in np.arange(Natomsy):
            i = xi+1
            j = yj+1
            parimparJ = divmod(j,2)
            parimparJf=parimparJ[1]
            parimparI = divmod(i+1,2)
            parimparIf=parimparI[1]
            x = (i-1)*np.sqrt(3)/2*cc
            y = (3*(j-1)+parimparJf+((-1)**(j))*parimparIf)*cc/2
            matxyz.append([y,x])
            xmat.append([y])
            ymat.append([x])
    return matxyz,xmat,ymat

def TB_HamArmchair_finite(width, length,U,t): 
    #t=1
    #U=0
    nsites = width*(2*length)
    
    #These are just in case we need to modify them later
    tside = t
    tabove = t

    #nsitesUC = 2*width #number of atoms in unit cell
    
    H0 = np.diag(U/2*np.ones(nsites,dtype="complex128"))#most of the time we will consider U=0
    
    abovejump = np.diag(tabove*np.ones(nsites-2*length,dtype="complex128"),k=2*length)
    
    sidejumpl = np.zeros(nsites-1,dtype="complex128")
    
    for k in range(len(sidejumpl)):
        if (1+k+(k//(2*length)))%2 and (k%(2*length)) != 2*length-1:
            sidejumpl[k] =tside
    
    sidejump= np.diag(sidejumpl,k=1)
    
    Htb= H0 + sidejump + abovejump
    
    Htb += Htb.transpose().conjugate()

    return Htb

def TB_HamArmchair_infinite(width,U,t,kx): #at Gamma point (k=0)
    #t=1
    #U=0
    H0 = np.diag(U/2*np.ones(2*width, dtype="complex128"))
    above = np.zeros(2*width-1)
    for i in range(len(above)):
        if i%4 == 0:
            above[i] += t
    jump = np.zeros(2*width-1)
    for i in range(len(jump)):
        if i%4 == 2:
            jump[i] += t
    sidejump = np.diag(jump, k=1)
    
    H0 += np.diag(above, k=1)
    H0 += np.diag(t*np.ones(2*width-2),k=2)
    
    H0 += sidejump*np.exp(1j*kx) #should be multiplied by cc but whatever
     
    
    
    H0 += H0.transpose().conjugate()
    return H0
    


if __name__ == "__main__":

    width = 25 #odd
    length = 10


    t = -1
    U = 0. #Code is designed for U=0
    cc=0.142

    kxs = np.linspace(-np.pi,np.pi,500)

    energies = []

    for kval in kxs:
        energies.append(eigvalsh(TB_HamArmchair_infinite(width,U,t,kval)))

    bands = np.sort(energies)
    halfgap = np.min(np.abs(bands))

    
    
    #np.savetxt("hamiltonian.txt",np.rint(TB_HamArmchair_finite(width, length,U,t)).astype(int), fmt='%.0e')
    eigvalsf, eigvect = eigh(TB_HamArmchair_finite(width, length,U,t))
    eigvalsf2, eigvect2 = eigh(TB_HamArmchair_finite(7, 100,U,t))
    #eigvalsinf = eigvalsh(TB_HamArmchair_infinite(width,U,t,0))
    eigvect = np.transpose(eigvect)
    
    
    
    g = 0.02
    res = 3000
    xvals = np.linspace(-3,3,res)
    dos = np.zeros(res)
    #dosinf = np.zeros(res)
    dosband = np.zeros(res)
    for eiv in eigvalsf:
        dos[:] += lorentzian(xvals[:],np.real(eiv),g)
        
    """for eivinf in eigvalsinf:
        dosinf[:] += lorentzian(xvals[:],eivinf,g)"""

    for band in bands:
        for eivband in band:
            dosband += lorentzian(xvals[:],np.real(eivband),g)

    #print(eigvals)

    '''plt.figure()
    plt.title("w={}".format(width))
    plt.subplot(121)
    plt.plot(kxs,bands, 'k')
    plt.ylabel("$\\varepsilon$ (units of $t$)")
    plt.xlabel("$k_x$")
    plt.subplot(122)
    plt.plot(dosband/np.max(dosband),xvals, 'k')'''
    
    fig = plt.figure(figsize=(10, 5))  # Adjust the figure size if needed
    fig.suptitle("w={}".format(width))

    ax1 = plt.subplot(121)
    ax1.plot(kxs, bands, 'k')
    ax1.set_ylabel("$\\varepsilon$ (units of $t$)")
    ax1.set_xlabel("$k_x$")

    ax2 = plt.subplot(122)
    ax2.plot(dosband/np.max(dosband), xvals, 'k')
    ax2.vlines(0,np.min(xvals), np.max(xvals),'r','dotted')
    ax2.set_yticks([])
    ax2.set_xticks([0,1])
    ax2.set_xlabel("DOS (normalized)")

    # Adjust the position and size of the subplots manually
    ax1.set_position([0.1, 0.1, 0.38, 0.8])  # [left, bottom, width, height]
    ax2.set_position([0.48, 0.1, 0.38, 0.8])

    

    plt.figure()
    plt.subplot(311)
    plt.vlines(U,0,1.02, 'r')
    plt.title("Finite. w = {}, l={}".format(width,length))
    #plt.xlabel("Energy (a.u.)")
    #plt.ylabel("Density of states (a.u.)")
    plt.plot(xvals,dos/np.max(dos))

    dos2 = np.zeros(res)
    for eiv2 in eigvalsf2:
        dos2[:] += lorentzian(xvals[:],np.real(eiv2),g)

    plt.subplot(312)
    plt.vlines(U,0,1.02, 'r')
    plt.title("Finite. w = 7, l=100".format(width,length))
    #plt.xlabel("Energy (a.u.)")
    plt.ylabel("Density of states (a.u.)")
    plt.plot(xvals,dos2/np.max(dos2))



    plt.subplot(313)
    plt.title("Infinite. w = {}".format(width))
    plt.xlabel("Energy (a.u.)")
    #plt.ylabel("Density of states (a.u.)")
    plt.plot(xvals,dosband/np.max(dosband), 'g')

    

    """plt.subplot(312)
    plt.title("Infinite. w = {} at k = 0".format(width))
    plt.plot(xvals,dosinf/np.max(dosinf), 'g')"""



    matxyz,xmat,ymat =  GrapheGENEArmchair(width,length,0.142)

    xlengths = np.linspace(-length,length,4*length)
    EdgestatesDensity = []
    #print(halfgap)
    #print(eigvect)
    edgestates = 0
    for l,(energy, state) in enumerate(zip(eigvalsf, eigvect)):
        if np.abs(energy) < halfgap:
            edgestates += 1
            denslength = np.zeros(4*length)
            print("Edge state with energy {}, index {}.".format(energy,l))
            state = state.reshape(width,2*length)
            for i in range(2*length):
                for j in range(2):
                    denslength[2*i+(i+(1-j))%2] = np.linalg.norm(state[j::2,i])**2
                    #print(len(state[j::2,i]))
                    #print(np.square(np.abs(state[(1-j)::,i]))) 
            EdgestatesDensity.append((denslength))#[:]+denslength[::-1])/2)
    #EdgestatesDensity = np.square(np.abs(eigvect[width*length-1]))
    print("Number of edge states = ", edgestates)



    plt.figure()
    indice=0
    plt.subplot(311)
    plt.scatter(xmat,ymat,25,c="r")  
    plt.title("w = {}, l={}".format(width,length))
    p =width*length-1-indice #TO DO: CHANGE THIS SO IT IS AUTOMATIC
    #print(eigvalsf[p])
    plt.scatter(xmat,ymat,s=np.square(np.abs(eigvect[p]))*100/np.max(np.square(np.abs(eigvect[p]))),c=np.square(np.abs(eigvect[p])))  
    plt.xlabel('x(nm)')
    plt.ylabel('y(nm)')
    plt.ylim(-0.1,cc*width)

    plt.subplot(312)
    plt.title("Density of electrons along the length of the ribbon")
    #plt.hlines(0,xlengths[0],xlengths[-1],r)
    for esd in EdgestatesDensity:
        plt.plot(xlengths,esd)
    plt.xlabel("Atom position (ribbon centered at 0)")
    
    plt.ylabel("Density of electrons (a.u.)")
    
    plt.subplot(313)
    plt.scatter(xmat,ymat,25,c="r")  
    plt.title("w = {}, l={}".format(width,length))
    p = width*length+indice
    #print(eigvalsf[p])
    plt.scatter(xmat,ymat,s=np.square(np.abs(eigvect[p]))*100/np.max(np.square(np.abs(eigvect[p]))),c=np.square(np.abs(eigvect[p])))  
    plt.xlabel('x(nm)')
    plt.ylabel('y(nm)')
    plt.ylim(-0.1,cc*width)
    
    
    plt.figure()
    plt.title("Density of electrons along the length of the ribbon")
    #plt.hlines(0,xlengths[0],xlengths[-1],r)
    for esd in EdgestatesDensity:
        plt.plot(xlengths,esd)
    plt.xlabel("Atom position (ribbon centered at 0)")
    
    plt.ylabel("Density of electrons (a.u.)")
    
    plt.show()
