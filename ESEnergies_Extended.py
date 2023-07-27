import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh, eigvalsh
from TB_GNR import TB_HamArmchair_finite, TB_HamArmchair_infinite
from scipy.optimize import curve_fit, fsolve
import warnings


def fk(kx,ky,t):
    return t*(1+np.exp(1j*kx)*2*np.cos(ky))  
    
def ftosolveq(qalphaytheta,ky,t,length):
    qalpha, theta = qalphaytheta
    kx = np.pi
    return [np.real(np.sinh((2*length)*qalpha + theta)), np.real(np.exp(2*theta)-(fk(kx-1j*qalpha,ky,t)/fk(kx+1j*qalpha,ky,t)))]


def edgestatesNOGAP_TB_AGNR(width, length, U,t): #width = 3p+1
    eigvals, eigvect = eigh(TB_HamArmchair_finite(width, length,U,t))
    eigvect = np.transpose(eigvect)
    p = int((width-1)/3)
    #eigvalsinf = eigvalsh(TB_HamArmchair_infinite(width,U,t,0)) #the gap is always at k=0
    #halfgap = np.min(np.abs(eigvalsinf))
    EdgestatesEnergies = []
    Nedgestates = 0
    ESdist = []
    nsites = width*2*length
    for l,(energy, state) in enumerate(zip(eigvals, eigvect)):
        if l >= (nsites/2-p/2) and l <= (nsites/2+p/2-1):
            Nedgestates += 1 
            EdgestatesEnergies.append(energy)
            ESdist.append(state)
    return Nedgestates, EdgestatesEnergies, ESdist


if __name__=='__main__':
    
    
    t = -1
    U = 0.
    cc=0.142
    
    
    

    widths = [31]#[7,13,19,25,31]
    lengths = np.arange(30)+1
    Nes = []
    for width in widths:
        ESEn = []
        nes = []
        for length in lengths:
            Nedgestates, EdgestatesEnergies, dummyvar = edgestatesNOGAP_TB_AGNR(width, length, U,t)
            ESEn.append(np.array(EdgestatesEnergies/np.abs(t)))
            nes.append(Nedgestates)

        
        Nes.append(nes)
        
        energies = []
        kxs = np.linspace(-np.pi,np.pi,500)
        for kval in kxs:
            energies.append(eigvalsh(TB_HamArmchair_infinite(width,U,t,kval)))

        bands = np.sort(energies)
        
        
        Figure = plt.figure()
        ax1 = plt.axes()
        #ax1 = plt.subplot(121)
        #ax1.plot(kxs, bands, 'k')
        #plt.ylim(0,np.abs(t))
        #plt.xlim(-np.pi,0)
        
        #ax2 = plt.subplot(122)
        plt.title("w={}".format(width))
        plt.xlabel("Length (unit cells)")
        plt.ylabel("Edge state energies (units of $t$)")
        #plt.ylim(0,np.abs(t))
        #plt.yscale('log')
        
        #ax1.set_position([0.1, 0.1, 0.38, 0.8])  # [left, bottom, width, height]
        #ax2.set_position([0.48, 0.1, 0.38, 0.8])

        for i in range(len(lengths)):
            for en in ESEn[i]:
                plt.scatter(lengths[i],np.abs(en),color='k')


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
            qalphadummy,thetadummy = fsolve(ftosolveq,[0.8,-10], args=(ky,t,dummylength),xtol=1e-5)
            #print(qalphadummy,thetadummy)
            qalphas.append(qalphadummy)
            #print(ftosolveq([qalphadummy,thetadummy],ky,t,dummylength))
        
        for i,ns in enumerate(nes):
            for n in range(int(ns/2)):
                lines[(ns//2-1)-n].append(np.array(ESEn[i])[-(1+n)])
        
        #plt.ylim(1e-7,np.max(np.abs(ESEn)))
        
        xs = np.linspace(1,lengths[-1]+1,1000)
             
        for j,line in enumerate(lines):
            skips = 0
            ilength = len(lengths) - len(line) + skips
            
            #popt, pcov = curve_fit(lambda t, a, b, c: a * np.exp(-width*b * t), lengths[ilength:], np.array(line[skips:]),p0=(np.max(np.array(line[skips]))+1, line[skips], 0), maxfev=20000)
            #popt, pcov = curve_fit(lambda t, a, b, c: 1 * np.sinh(b)/np.sinh((2*t+1)*b), lengths[ilength:], np.array(line[skips:]),p0=(1, 0.01, 0), maxfev=20000)
            
            
            #print(popt)
            
            #a = popt[0]
            #b = popt[1]
            #phi = LocalizationLengthofES(width,j,t,U,cc, length=25,iguess=3*cc/2/b)
            
            #print(j,qalphas[j],b)
            
            
            
            #plt.plot(xs, 1*np.sinh(b)/np.sinh((2*xs+1)*b))
            #plt.plot(xs, -1*np.sinh(b)/np.sinh((2*xs+1)*b))
            plt.plot(xs, 1*np.sinh(qalphas[j])/np.sinh((2*xs+1)*qalphas[j]),'--', color='C{}'.format(10-1-j))
            #plt.plot(xs, -1*np.sinh(qalphas[j])/np.sinh((2*xs+1)*qalphas[j]),'--')
    

        sizew,sizel = Figure.get_size_inches()/2
        ax2 = ax1.inset_axes([0.3,0.3,0.69,0.69])
        #ax2.set_position()
        ax2.set_zorder(10)
        ax2.set_yscale('log')
        ax2.set_ylim(1e-3,1)
        for i in range(len(lengths)):
            for en in ESEn[i]:
                ax2.scatter(lengths[i],np.abs(en),color='k')
                
        for j,line in enumerate(lines):
            ax2.plot(xs, 1*np.sinh(qalphas[j])/np.sinh((2*xs+1)*qalphas[j]),'--', color='C{}'.format(10-1-j))
            

    plt.show()