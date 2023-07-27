import numpy as np
import matplotlib.pyplot as plt




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
            xmat.append(x)
            ymat.append(y)
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
            xmat.append(y)
            ymat.append(x)
    return xmat,ymat


def GrapheGENEHaiku(length,cc, closedEdges = True):  
    NUCs = length #times the unit cell repeats along the ribbon

    xmat = []
    ymat = []
    
    d = (3/2*cc)
    edge=0
    UC = [] #unit cell

    UCxmat = []
    UCymat= []
    
    dummyxmat, dummyymat = GrapheGENEArmchair(5,1,cc)
    
    
    if closedEdges:
        edge = d
        xmat = np.append(xmat, np.delete(dummyxmat,[1,3,5,7,9]))
        ymat = np.append(ymat, np.delete(dummyymat,[1,3,5,7,9]))
    
    
    UCxmat = np.append(UCxmat, np.delete(dummyxmat,[0,2,4,6,8])+(-1)*d)
    UCymat = np.append(UCymat, np.delete(dummyymat,[0,2,4,6,8]))
    

    
    UCxmat = np.append(UCxmat, np.delete(dummyxmat,[1,3,5,7,9])+(1)*d)
    UCymat = np.append(UCymat, np.delete(dummyymat,[1,3,5,7,9]))
    
    dummyUCxmat = np.copy(UCxmat)
    dummyUCymat = np.copy(UCymat)
    
    dummy7x, dummy7y = GrapheGENEArmchair(7,1,cc)
    
    UCxmat = np.append(UCxmat, dummy7x[::2]+((2)*d)*np.ones(len(dummy7x)//2))
    UCymat = np.append(UCymat, dummy7y[::2]-np.sqrt(3)/2*cc*np.ones(len(dummy7y)//2))
    
    UCxmat = np.append(UCxmat, dummy7x[1::2]+((2)*d)*np.ones(len(dummy7x)//2))
    UCymat = np.append(UCymat, dummy7y[1::2]-np.sqrt(3)/2*cc*np.ones(len(dummy7y)//2))
    
    UCxmat = np.append(UCxmat, dummyUCxmat.flatten()+((4)*d)*np.ones(len(dummyUCxmat.flatten())))
    UCymat = np.append(UCymat, dummyUCymat)
    
    for l in range(NUCs):
        xmat = np.append(xmat,UCxmat+l*(6)*d+edge)
        ymat = np.append(ymat,UCymat)
    
    if closedEdges:
        xmat = np.append(xmat, np.delete(dummyxmat,[0,2,4,6,8])+NUCs*(6)*d)
        ymat = np.append(ymat, np.delete(dummyymat,[0,2,4,6,8]))  
    
    return np.transpose(np.array([xmat,ymat])),xmat,ymat

if __name__=='__main__':
    width = 13 #odd
    length = 5 #odd

    xmat,ymat =  GrapheGENEArmchair(width,length,0.142*10) #It is designed for width odd

    centered = True

    with open('prueba.xyz', 'w') as f:
        f.write(str(len(xmat))+"\n")
        f.write("AGNR width={} and length={}\n".format(width,length))
        for x, y in zip(xmat,ymat):
            f.write("C  {:024.20f}  {:024.20f}   0\n".format(x,y))

    width2 = 7
    length2 = 5
    xmat2,ymat2 =  GrapheGENEArmchair(width2,length2,0.142*10)



    yvalue = 0
    if centered:
        for i in range(width2)[::-1]:
            xmat2 = np.delete(xmat2,2*length2*(i))
            ymat2 = np.delete(ymat2,2*length2*(i))
        
        yvalue = (np.max(ymat)+np.min(ymat))/2 - (np.max(ymat2)+np.min(ymat2))/2 

    xmat3 = np.append(xmat,xmat2-np.min(xmat2)+np.max(xmat)+0.142*10)
    ymat3 = np.append(ymat,np.array(ymat2)+yvalue)

    with open('pruebajunction.xyz', 'w') as f:
        f.write(str(len(xmat3))+"\n")
        f.write("AGNR width={} and length={}\n".format(width2,length2))
        for x, y in zip(xmat3,ymat3):
            f.write("C  {:024.20f}  {:024.20f}   0\n".format(x,y))

    length=5

    matxyz,xmat,ymat =  GrapheGENEHaiku(length,0.142*10)#,closedEdges = False) 



    with open('Haikuprueba.xyz', 'w') as f:
        f.write(str(len(xmat))+"\n")
        f.write("Haiku-AGNR of length={}\n".format(length))
        for x, y in zip(xmat,ymat):
            f.write("C  {:024.20f}  {:024.20f}   0\n".format(x,y))
            

    plt.figure(1)
    plt.scatter(xmat,ymat,50,c="r")  
    plt.xlabel('x(nm)')
    plt.ylabel('y(nm)')

    plt.draw()
    plt.show() 