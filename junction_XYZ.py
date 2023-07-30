import numpy as np
from TB_GNR import GrapheGENEArmchair



def junction_xyz(width1,length1,width2,length2,cc, centered = False, three = False):
    

    matxyz1, xmat1,ymat1 =  GrapheGENEArmchair(width1,length1,cc)
    
    
    matxyz2, xmat2,ymat2 =  GrapheGENEArmchair(width2,length2,cc)

    impar = abs((width1-1)/6 - (width2-1)/6)%2

    yvalue = 0
    if centered:
        if impar:
            for i in range(width2)[::-1]:
                xmat2 = np.delete(xmat2,2*length2*(i))
                ymat2 = np.delete(ymat2,2*length2*(i))
        else:
            pass
        
        yvalue = (np.max(ymat1)+np.min(ymat1))/2 - (np.max(ymat2)+np.min(ymat2))/2 

    xmat3 = np.append(xmat1,xmat2-np.min(xmat2)+np.max(xmat1)+cc)
    ymat3 = np.append(ymat1,np.array(ymat2)+yvalue)
    
    if three:
        yvalue = 0
        if centered:
            if impar:
                for i in range(width2)[::-1]:
                    xmat3 = np.delete(xmat3,(2*length2-1)*(i+1)+2*width1*length1-1)
                    ymat3 = np.delete(ymat3,(2*length2-1)*(i+1)+2*width1*length1-1)
                    
            else:
                pass
         
        
        xmat3 = np.append(xmat3,xmat1-np.min(xmat1)+np.max(xmat3)+cc)
        ymat3 = np.append(ymat3,np.array(ymat1)+yvalue)
        
    
    return np.transpose(np.array([xmat3,ymat3])), xmat3, ymat3



if __name__=='__main__':
    cc = 0.142
    width1 =7 
    length1 = 10 
    
    width2 = 7
    length2 = 10
    
    matxyz, xmat, ymat = junction_xyz(width1,length1,width2,length2,cc*10, centered = False, three=True)
    with open('pruebajunction2.xyz', 'w') as f:
        f.write(str(len(xmat))+"\n")
        f.write("AGNR with a junction\n")
        for x, y in zip(xmat,ymat):
            f.write("C  {:024.20f}  {:024.20f}   0\n".format(x,y))

    
    