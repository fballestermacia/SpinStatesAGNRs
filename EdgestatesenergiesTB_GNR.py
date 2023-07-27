import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, eigvalsh
from TB_GNR import TB_HamArmchair_finite, TB_HamArmchair_infinite
from scipy.optimize import curve_fit, fsolve
import warnings

warnings.filterwarnings('ignore')


def edgestates_TB_AGNR(width, length, U,t):
    eigvals, eigvect = eigh(TB_HamArmchair_finite(width, length,U,t))
    eigvect = np.transpose(eigvect)
    eigvalsinf = eigvalsh(TB_HamArmchair_infinite(width,U,t,0)) #the gap is always at k=0
    halfgap = np.min(np.abs(eigvalsinf))
    EdgestatesEnergies = []
    Nedgestates = 0
    ESdist = []
    if length ==1:
        Nedgestates += 1
        EdgestatesEnergies.append(eigvals[width*length-1])
        EdgestatesEnergies.append(eigvals[width*length])
    for l,(energy, state) in enumerate(zip(eigvals, eigvect)):
        if np.abs(energy) <halfgap:
            Nedgestates += 1 
            EdgestatesEnergies.append(energy)
            ESdist.append(state)
    return Nedgestates, EdgestatesEnergies, ESdist


def LocalizationLengthofES(width,jj,t,U,cc, length=25,iguess=1):
    Nedgestates, EdgestatesEnergies, ESCoeffs = edgestates_TB_AGNR(width, length, U,t)
    
    p1 = jj
    p2 = (-jj-1)
    
    if (width == 25 and (jj==3 or jj==0)):
        leftES = (np.real(ESCoeffs[p1].flatten()) + np.real(ESCoeffs[p2].flatten()))[::-1]
    else:
        leftES = np.real(ESCoeffs[p1].flatten()) + np.real(ESCoeffs[p2].flatten())
    #leftES=np.real(ESCoeffs[p2].flatten())
    denslength = np.zeros(4*length)
    leftES = leftES.reshape(width,2*length)
    for i in range(2*length):
        for j in range(2):
            denslength[2*i+(i+(1-j))%2] = np.linalg.norm(leftES[j::2,i])**2
    popt, pcov = curve_fit(lambda t, a, b: a * np.exp(-b * t), np.linspace(0,length,length)*cc, denslength[:len(denslength)//2:2], p0=(denslength[0],iguess), maxfev=20000)
    a = popt[0]
    b = popt[1]
    phi=3*cc/2/b
    
    '''plt.figure()
    plt.plot(np.linspace(0,2*length,4*length),denslength)
    plt.plot(np.linspace(0,2*length,4*length),a * np.exp(-b *cc*np.linspace(0,2*length,4*length)) )
    '''
    return phi
    

def fk(kx,ky,t):
    return t*(1+np.exp(1j*kx)*2*np.cos(ky))  
    
def ftosolveq(qalphaytheta,ky,t,length):
    qalpha, theta = qalphaytheta
    kx = np.pi
    return [np.real(np.sinh((2*length)*qalpha + theta)), np.real(np.exp(2*theta)-(fk(kx-1j*qalpha,ky,t)/fk(kx+1j*qalpha,ky,t)))]


if __name__ == "__main__":
    #width = 7 #odd
    #length = 100
    t = -1
    U = 0.
    cc=0.142
    
    
    

    widths = [25]#[7,13,19,25,31]
    lengths = np.arange(40)+1
    Nes = []
    for width in widths:
        ESEn = []
        nes = []
        for length in lengths:
            Nedgestates, EdgestatesEnergies, dummyvar = edgestates_TB_AGNR(width, length, U,t)
            ESEn.append(np.array(EdgestatesEnergies/np.abs(t)))
            nes.append(Nedgestates)

        
        Nes.append(nes)
        
        
        plt.figure()
        plt.title("w={}".format(width))
        plt.xlabel("Length")
        plt.ylabel("Edge state energies within gap (a.u.)")
        #plt.yscale('log')

        for i in range(len(lengths)):
            for en in ESEn[i]:
                plt.scatter(lengths[i],en,color='k')


        lines = []
        for j in range(np.max(nes)//2):
            lines.append([])
        
        qalphas = []
        alphas = np.arange((width+1)/2)+1
        #print(alphas[-2:-np.max(nes)//2-2:-1])
        kys = np.pi/(width+1)*alphas[-2:-np.max(nes)//2-2:-1]
        #print(kys)
        dummylength = 100
        for ky in kys:
            qalphadummy,thetadummy = fsolve(ftosolveq,[0.8,-10], args=(ky,t,dummylength),xtol=1e-10)
            #print(qalphadummy,thetadummy)
            qalphas.append(qalphadummy)
            #print(ftosolveq([qalphadummy,thetadummy],ky,t,dummylength))
        
        for i,ns in enumerate(nes):
            for n in range(int(ns/2)):
                lines[(ns//2-1)-n].append(np.array(ESEn[i])[-(1+n)])
        
              
        for j,line in enumerate(lines):
            skips = 0
            ilength = len(lengths) - len(line) + skips
            
            #popt, pcov = curve_fit(lambda t, a, b, c: a * np.exp(-width*b * t), lengths[ilength:], np.array(line[skips:]),p0=(np.max(np.array(line[skips]))+1, line[skips], 0), maxfev=20000)
            popt, pcov = curve_fit(lambda t, a, b, c: 1 * np.sinh(b)/np.sinh((2*t+1)*b), lengths[ilength:], np.array(line[skips:]),p0=(1, 0.01, 0), maxfev=20000)
            
            
            #print(popt)
            
            a = popt[0]
            b = popt[1]
            #phi = LocalizationLengthofES(width,j,t,U,cc, length=25,iguess=3*cc/2/b)
            
            print(j,qalphas[j],b)
            
            xs = np.linspace(1,lengths[-1]+1,100)
            
            plt.plot(xs, 1*np.sinh(b)/np.sinh((2*xs+1)*b))
            plt.plot(xs, -1*np.sinh(b)/np.sinh((2*xs+1)*b))
            plt.plot(xs, 1*np.sinh(qalphas[j])/np.sinh((2*xs+1)*qalphas[j]),'--')
            plt.plot(xs, -1*np.sinh(qalphas[j])/np.sinh((2*xs+1)*qalphas[j]),'--')
            
            

    plt.figure()
    plt.ylabel("Number of Edge States within gap")
    plt.xlabel("Length")
    for ness, width in zip(Nes,widths):
        plt.plot(lengths,ness, label="w={}".format(width))
    plt.legend()

    plt.show()
            

