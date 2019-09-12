import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.collections as mcoll

from mpl_toolkits.mplot3d import Axes3D

from copy import deepcopy as dp

import math

from funnel_package.utils import *
## Create list of line segments from x and y coordinates, in the correct format 
# for LineCollection: an array of the form numlines x (points per line) x 2 (x
# and y) array
def make_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

def getGradLine(x,y,cmap, z=None, linewidth=2.0):
    x=np.array(x); x.reshape((x.size,));
    y=np.array(y); y.reshape((y.size,));
    if z==None:
        z = np.arange(0,x.size,1)
    z=np.array(z); z.reshape((z.size,));
    assert x.size==y.size==z.size, 'Expecting row vectors of same lengths; z can also be None'
    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, linewidth=linewidth)
    return lc

def plotGradLine(ax,x,y,cmap, z=None, linewidth=2.0):
    lc = getGradLine(x, y, cmap, z, linewidth)
    ax.add_collection(lc)
    return 0

nPointsEllip = 0
if nPointsEllip > 0:
    uVec2 = np.array([1.0,0.0]).reshape((2,1))
    allAng = np.arange(0.0, (2-1/nPointsEllip)*np.pi, 2*np.pi/nPointsEllip)
    Rot = lambda aAng: np.array([[np.cos(aAng), -np.sin(aAng)], [np.sin(aAng), np.cos(aAng)]])
    allUvec2 = np.zeros((2,nPointsEllip))
    for k in range(nPointsEllip):
        allUvec2[:,k] = dot(Rot(allAng[k]), uVec2).squeeze()
else:
    allUvec2 = np.zeros((2,0))

def getEllipse(pos,P,alpha):
    E, v = eig(P)
    orient = math.atan2(E[1,0],E[0,0])*180.0/np.pi
    return Ellipse(xy=pos, height=2.0*math.sqrt(alpha) * 1.0/math.sqrt(v[1]), width=2.0*math.sqrt(alpha) * 1.0/math.sqrt(v[0]), angle=orient)

def plotEllipse(ax, pos, P, alpha, color = [0.0,0.0,1.0,1.0], faceAlpha=0.5, doPoints = False):
    color=np.array(dp(color)); color[-1]=color[-1]*faceAlpha; color=list(color)
    e = getEllipse(pos, P, alpha)
    ax.add_patch(e)
    e.set_facecolor( color )
    if doPoints:
        T = getT(P)*alpha**0.5
        Ps = np.zeros(allUvec2.shape)
        for k in range(allUvec2.shape[1]):
            Ps[:,k] = dot(T, allUvec2[:,k].reshape((2,1))).squeeze()
        mid = np.array(e.center).reshape((2,1))
        Ps= Ps+mid
        ax.plot(Ps[0,:], Ps[1,:], '.', color=color)
    return 0
    
    
def plotTransition(ax, aTimedAut, aTrans, C1_p, C2_p, dim=[0,1], color = [[0.0,0.0,1.0,1.0], [1.0,0.0,0.0,1.0]], deltaT = 3.0, faceAlpha=0.5):
    #Plot the transition aTrans defined within the timed automata aTimedAut in the given axis
    #Ellipse color -> set face alpha
    eColor = []
    for aCol in color:
        aCol=np.array(dp(aCol)); aCol[-1]=aCol[-1]*faceAlpha; aCol=list(aCol)
        eColor.append(aCol)
     
    #two structures
    p = astruct()
    c = astruct()
    #Get concerned funnels
    p.fun = aTimedAut.funnelDict[aTrans.parentID]
    c.fun = aTimedAut.funnelDict[aTrans.childID]
    #Get the ellipse in the plotting dimensions
    p.Psd = p.fun.dynSys.getP()[np.ix_(dim, dim)]
    p.Tpsd = getT(p.Psd, p.fun.lyapVal)
    c.Psd = c.fun.dynSys.getP()[np.ix_(dim, dim)]
    c.Tpsd = getT(c.Psd, c.fun.lyapVal)
    #Get times
    C1_c, _unused = myExec(aTrans.transSet, C1_p, C2_p)
    
    dT = aTrans.tSpan[1]-aTrans.tSpan[0]
    #Take care of times
    p.T = guround( np.linspace(max(aTrans.tSpan[0]-dT*deltaT, p.fun.tSpanD[0]), min(aTrans.tSpan[1]+dT*deltaT, p.fun.tSpanD[1]), p.fun.traj.N_searchFrom) )
    c.T = guround( np.linspace(max(C1_c-dT*deltaT, c.fun.tSpanD[0]), min(C1_c+dT*deltaT, c.fun.tSpanD[1]), p.fun.traj.N_searchFrom) )
    p.X = p.fun.traj.getX(p.T)
    p.X = p.X[dim, :]
    c.X = c.fun.traj.getX(c.T)
    c.X = c.X[dim, :]
    #Create ellipsoids
    p.E = [getEllipse(p.fun.traj.getX( guround(aTrans.tSpan[0]) )[dim,:], p.Psd, p.fun.lyapVal), getEllipse(p.fun.traj.getX( guround(C1_p) )[dim,:], p.Psd, p.fun.lyapVal), getEllipse(p.fun.traj.getX( guround(aTrans.tSpan[1]) )[dim,:], p.Psd, p.fun.lyapVal) ]     
    c.E = [getEllipse(c.fun.traj.getX( guround(C1_c) )[dim,:], c.Psd, c.fun.lyapVal)]
    
    #Add points to facilitate scaling
    p.Ep = np.zeros(allUvec2.shape)
    c.Ep = np.zeros(allUvec2.shape)
    for k in range(allUvec2.shape[1]):
        p.Ep[:,k] = dot(p.Tpsd, allUvec2[:,k].reshape((2,1))).squeeze()
        c.Ep[:,k] = dot(c.Tpsd, allUvec2[:,k].reshape((2,1))).squeeze()
    
    #Plot everything
    ax.plot(p.X[0,:], p.X[1,:], color=color[0])
    ax.plot(c.X[0,:], c.X[1,:], color=color[1])
    for ellip in p.E:
        ax.add_patch(ellip)
        ellip.set_facecolor( eColor[0] )
        mid = np.array(ellip.center).reshape((2,1))
        Ep = p.Ep+mid
        ax.plot(Ep[0,:], Ep[1,:], '.', color=color[0])
    for ellip in c.E:
        ax.add_patch(ellip)
        ellip.set_facecolor( eColor[1] )
        mid = np.array(ellip.center).reshape((2,1))
        Ep = c.Ep+mid
        ax.plot(Ep[0,:], Ep[1,:], '.', color=color[1])  

    #done
    return 0
#pu.plotTransition2(ax[0], self, pFun, C1_p, cFun, C1_c, color=[cList[cNum], cList[cNum+1]], faceAlpha=faceAlpha)
def plotTransition2(ax, aTimedAut, pFun, C1_p, cFun, C1_c, dim=[0,1], color = [[0.0,0.0,1.0,1.0], [1.0,0.0,0.0,1.0]], deltaT = 3.0, faceAlpha=0.5):
    #Plot the transition aTrans defined within the timed automata aTimedAut in the given axis
    #Ellipse color -> set face alpha
    eColor = []
    for aCol in color:
        aCol=np.array(dp(aCol)); aCol[-1]=aCol[-1]*faceAlpha; aCol=list(aCol)
        eColor.append(aCol)
     
    #two structures
    p = astruct()
    c = astruct()
    #Get concerned funnels
    p.fun = pFun
    c.fun = cFun
    #Get the ellipse in the plotting dimensions
    p.Psd = p.fun.dynSys.getP()[np.ix_(dim, dim)]
    p.Tpsd = getT(p.Psd, p.fun.lyapVal)
    c.Psd = c.fun.dynSys.getP()[np.ix_(dim, dim)]
    c.Tpsd = getT(c.Psd, c.fun.lyapVal)
    
    #Create ellipsoids
    #Clocks are in continuous real time
    p.E = getEllipse(p.fun.traj.getX(C1_p)[dim,:], p.Psd, p.fun.lyapVal)     
    c.E = getEllipse(c.fun.traj.getX(C1_c)[dim,:], c.Psd, c.fun.lyapVal)
    
    #Add points to facilitate scaling
    p.Ep = np.zeros(allUvec2.shape)
    c.Ep = np.zeros(allUvec2.shape)
    for k in range(allUvec2.shape[1]):
        p.Ep[:,k] = dot(p.Tpsd, allUvec2[:,k].reshape((2,1))).squeeze()
        c.Ep[:,k] = dot(c.Tpsd, allUvec2[:,k].reshape((2,1))).squeeze()
    
    #Plot ellipsoids
    ax.add_patch(p.E)
    p.E.set_facecolor( eColor[0] )
    mid = np.array(p.E.center).reshape((2,1))
    Ep = p.Ep+mid
    ax.plot(Ep[0,:], Ep[1,:], '.', color=color[0])
    ax.add_patch(c.E)
    c.E.set_facecolor( eColor[0] )
    mid = np.array(c.E.center).reshape((2,1))
    Ep = c.Ep+mid
    ax.plot(Ep[0,:], Ep[1,:], '.', color=color[1])
        
    #done
    return 0


def getColorList(numCol, cMap='jet'):
    #Create a predefined list of rgba colors
    jet = cm = plt.get_cmap(cMap) 
    cNorm  = colors.Normalize(vmin=0, vmax=(numCol-1))
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
    cList = []
    for k in range(numCol):
        cList.append(scalarMap.to_rgba(k))
    return cList
    

## This works on continuous time
def plotAllTrans(aTimedAut, dim=[0,1], cMap='jet', faceAlpha=0.5, deltaT = 3.0, fromFun=None, toFun=None, tStart=-np.Inf, tEnd=np.Inf, allInOne = False):
    #Plots all transitions possible in a given timed automata. each one in a seperate subplot
    #Collect all transitions and funnels
    allTrans = []
    funColorDict = {}
    funIDS = list(aTimedAut.funnelDict.keys())
    nFun = len(funIDS)
    cList = getColorList(nFun, cMap)
    
    for k in range(nFun):
        #getColor
        aID = funIDS[k]
        funColorDict[aID] = cList[k]
        
        #Get all transitions
        for aTrans in aTimedAut.funnelDict[aID].startIn2([tStart, tEnd]):
            #aTransList = aTimedAut.funnelDict[aID].transDict[aKey]
            #for aTrans in aTransList:
            #    allTrans.append(aTrans)
            check = True
            if fromFun!=None: check = check and aTrans.parentID in fromFun
            if toFun!=None: check = check and aTrans.childID in toFun
            if check: allTrans.append(aTrans)
    
    #Set up the figure and do the plot
    if allInOne:
        fid,axarr = plt.subplots(1, figsize=(15.0,9.0))
        
        for aTrans in allTrans:
            C1_p = 0.5*(aTrans.tSpan[0]+aTrans.tSpan[1])
            C2_p = 0.0
            #TBD find a less creepy way to do this
            print( aTrans.parentID+'->'+aTrans.childID+': '+aTrans.transSetPLOT + '\n'+aTrans.transGuardPLOT )
            plotTransition(axarr, aTimedAut, aTrans, C1_p, C2_p, dim=dim, color = [funColorDict[aTrans.parentID], funColorDict[aTrans.childID]], deltaT=deltaT, faceAlpha=faceAlpha)
        
    else:
        nR=0;nC=0;
        nSP = len(allTrans)
        if nSP == 0:
            print('No corresponding transitions found; Skipping plot \n')
            return 1
        while True:
            nC+=1
            if nR*nC>=nSP: break
            nR+=1
            if nR*nC>=nSP: break
    
        fid,axarr = plt.subplots(nR,nC, figsize=(15.0,9.0))
    
        for k in range(nR):
            for l in range(nC):
                if nR==1 and nC==1:
                    thisAx = axarr
                elif nR==1 and nC!=1:
                    thisAx = axarr[l]
                else:
                    thisAx = axarr[k,l]   
                ind = k*nC+l
                if ind >= nSP: break
                aTrans = allTrans[ind]
                C1_p = 0.5*(aTrans.tSpan[0]+aTrans.tSpan[1])
                C2_p = 0.0
                #TBD find a less creepy way to do this
                thisAx.set_title( aTrans.parentID+'->'+aTrans.childID+': '+aTrans.transSetPLOT + '\n'+aTrans.transGuardPLOT )
                plotTransition(thisAx, aTimedAut, aTrans, C1_p, C2_p, dim=dim, color = [funColorDict[aTrans.parentID], funColorDict[aTrans.childID]], deltaT=deltaT, faceAlpha=faceAlpha)
            
    #Done
    return 0


## Returns a ellipse shaped meshgrid
def ellipseVectorField(ax, Abar, P, alpha, color = [0.0,0.0,1.0,1.0], numSize=5, numPoints=20, dim = [0,1], doStreamLines=True, **kwargs):
    P = P[np.ix_(dim,dim)]
    Tu = getT(P)
    alphaLevels = np.linspace((alpha**0.5)/numSize, alpha**0.5, numSize)
    orientPoints =  np.int_( np.ceil( (np.linspace(0.3*numPoints,numPoints,numSize)**2.0)/(numPoints) ))
    
    num = orientPoints.sum()
    outPos = np.zeros((2,num))
    outVec = np.zeros((2,num))
    i=0
    
    m = Abar.shape[1]
    
    for k in range(numSize):
        thisAlpha = alphaLevels[k]
        thisPoints = orientPoints[k]
        thisAllAng = np.arange(0.0, (2-1/thisPoints)*np.pi, 2*np.pi/thisPoints)
        for l in range(thisPoints):
            thisVec = thisAlpha*ndot( Tu, Rot(thisAllAng[l]), uVec2 )
            outPos[:,i] = thisVec.squeeze()
            zVec = np.zeros((m,1))
            zVec[dim]=thisVec
            outVec[:,i] = (dot(Abar, zVec)[dim]).squeeze()
            i=i+1
        if k == numSize-1 and doStreamLines:
            _unused, e = eig(Abar, discardImag=False)
            er = np.min(np.abs(np.real(e)))
            ei = np.max(np.abs(np.imag(e)))
            if ei >=1e-3:
                t = np.arange(0.0,3*1/er, 1/20*1/ei)
                deltaP = np.int_(np.ceil(2*(ei**0.5/er**0.5)))
                deltaP = max(1, deltaP)
            else:
                t = np.linspace(0.0, 2*1/er, 300)
                deltaP=1
            lines = []
            for l in range(0,thisPoints, deltaP):
                thisVec=np.zeros((m,1))
                thisVec[dim,:]=outPos[:,num-1-l].reshape((m,1))
                t=t.reshape((1,t.size))
                func = lambda aT: dot(expm(Abar*aT),thisVec).squeeze()
                lines.append( np.apply_along_axis(func, 0, t)[dim,:] )
    
    if ax != None:
        if doStreamLines:
            for line in lines:
                ax.plot(line[0,:], line[1,:], color = color, linewidth=1.5)
        if not ('color' in kwargs.keys()):kwargs['color'] = color
        if not ('pivot' in kwargs.keys()):kwargs['pivot']='middle'
        if not ('width' in kwargs.keys()):kwargs['width']=0.005
        if not ('angles' in kwargs.keys()):kwargs['angles']='xy'
        ax.quiver(outPos[0,:], outPos[1,:], outVec[0,:], outVec[1,:], **kwargs)
    return [outPos, outVec]