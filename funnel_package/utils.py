## @package utils
# Implements useful functions related to funnel construction
# Within the classes and functions implemented, take is taken to ensure consistency when discretized

import numpy as np
from numpy import dot as dot
from numpy import log2 as ln
from numpy import log10 as log
from numpy import concatenate as concat
from numpy import multiply as mult
from numpy import divide as div
from numpy import conjugate as conj
from numpy import sqrt as sqrt 
from numpy import absolute as nabs
from numpy import minimum as nmin
from numpy import maximum as nmax
from numpy import amin as namin
from numpy import amax as namax
from numpy import diag as diag
from numpy.linalg import norm as norm
import scipy as sp
from scipy.linalg import svd as svd
from scipy.linalg import eig as seig
from scipy.linalg import eig as seigh
from scipy.linalg import cholesky as chol #Upper cholesky decomposition
from scipy.linalg import inv as inv
from scipy.interpolate import Akima1DInterpolator as int1D

from copy import deepcopy as dp

from scipy.linalg import solve_lyapunov
from scipy.linalg import expm as expm

import warnings
warnings.simplefilter('always', UserWarning)
#Global rounding value
globalRoundVal = 1e4
globalRoundVal2 = globalRoundVal//2
#Modify the granularity
#globalRoundVal = 3600
#globalRoundVal2 = 600
intInf = np.int_(1e18)
floatInf = np.float_(1e18)

## Provide a 1D interpolation utility
def doSampleInter(Xsample, Xorig, *args):
    #Make unique
    Xorig, IndOrig = dp( np.unique(Xorig, return_index=True) )
    outList = [Xsample]
    for aYarr in args:
        outList.append( np.zeros((aYarr.shape[0], Xsample.size)) )
        for k in range(aYarr.shape[0]):
            thisInter = int1D( Xorig, aYarr[k,IndOrig] )
            outList[-1][k,:] = thisInter(Xsample)
    return outList

#Round
def ground(inTime):
    inTime = np.array(inTime); inTime = inTime.reshape((inTime.size,))
    ind = np.where(np.isinf(inTime))
    isign = np.sign(inTime[ind])
    inTime[ind] = 0
    inTime = np.int_(np.rint(inTime*globalRoundVal)).reshape((inTime.size,))
    inTime[ind]  = isign*intInf
    return inTime
## unround/scale the internally used integers
def guround(inTime):
    inTime = np.array(inTime); inTime = inTime.reshape((inTime.size,))
    return (np.float_(inTime)/globalRoundVal).reshape((inTime.size,))
def gToNearest(inTime):
    return guround(ground(inTime))
#Round
def ground2(inTime):
    inTime = np.array(inTime); inTime = inTime.reshape((inTime.size,))
    ind = np.where(np.isinf(inTime))
    isign = np.sign(inTime[ind])
    inTime[ind] = 0
    inTime = np.int_( np.rint(inTime*globalRoundVal2)).reshape((inTime.size,))
    inTime[ind]  = isign*intInf
    return inTime
## unround/scale the internally used integers
def guround2(inTime):
    inTime = np.array(inTime); inTime = inTime.reshape((inTime.size,))
    return (np.float_(inTime)/globalRoundVal2).reshape((inTime.size,))
def gToNearest2(inTime):
    return guround2(ground2(inTime))


## vdc grants easy access to points in a one dimensional van der corput sequence
#
# In difference to the general van der corput sequence, 0 and 1 are returned at the beginning.
# @page vdc
class vdc(object):
    ## Constructor
    # @param N[integer] length of precomputed van der corput sequence, defaulted to 10000
    # @param base[float] The base of the van der corput sequence determines the distribution of the points. Defaulted to binary base.  
    def __init__(self, N=10000, base=2.0, current = 0):
        self.list=[1.0]
        self.current = current
        for k in range(N):
            n=k
            avdc, denom = 0,1
            while n:
                denom *= base
                n, remainder = divmod(n, base)
                avdc += remainder / denom
            self.list.append(avdc)
        self.list = np.array(self.list)
    ## Returns the next element in the list or empty list if the list way already entirely returned
    # @returns float
    def pop(self):
        if self.current == len(self.list)-1:
            self = vdc(2*int(len(self.list)), base = self.base, current = self.current)
        self.current+=1
        return float(self.list[self.current-1])
    ## Returns the subsequence up to the Nout'th element.
    # @param Nout[integer] length of returned subsequence
    # @returns np.array with (Nout,) as shape
    def getN(self, Nout):
        if Nout > self.list.size:
            self = vdc(Nout, base = self.base, current = self.current)
        return self.list[0:Nout]

globVDC = vdc()

## Initializes an empty object that behaves similar to a matlab struct
# @returns self[object] empty struct
class astruct(object):
    ## Sets all key/value pairs as writeable properties into the structure
    # @param aDict[dict] Standar Python dictionnary to be conversted into a structure
    def fromDict(self,  aDict):
        for (akey,  avalue) in aDict.items():
            setattr(self,  akey,  avalue)
    ## Implements the standard python get functionality used by dictionaries
    # @param aAttr[string] Name of the attribute to be read as string
    # @param defVal[~] Default value in case the attribute does not exist in the structure
    # @return [~] Content of the attribute or defVal
    def get(self,  aAttr,  defVal):
        if hasattr(self,  aAttr):
            return getattr(self,  aAttr)
        else:
            return defVal

## Convenience class to generate a structure holding QR matrices to be used (among others) as input Qstructorlist in @ref funnel1
# @page QRstuct
class QRstruct(object):
    ## Constructor
    # @param QIn[np.array] Q matrix
    # @param RIn[np.array] R matrix
    def __init__(self, QIn=None, RIn=None):
        super(QRstruct, self).__init__()
        self.Q = QIn
        self.R = RIn

## Matlab style find(x, 'first') 
#
# function returning the index of the first True value found in the list
# @param aIter[iterable over booleans]
# @return [integer] Index of first True value
def ffind(aIter):
    return np.flatnonzero(aIter)[0]
## Matlab style find(x, 'last') function 
#
# function returning the index of the first True value found in the list
# @param aIter[iterable over booleans]
# @return [integer] Index of last True value
def lfind(aIter):
    return np.flatnonzero(aIter)[-1]

## Matalb style eig function
#
# Eigenvector/values decomposition. By default only real eigenvalues are accepted.
# @param A[np.array] Matrix to be decomposed
# @param sort[string; None] 'ascending' or 'descending' if sorting is desired; None if unsorted return (default)
# @param atype[string; None] 'normed' if the norm of complex eigenvalues shall be considered; None if eigenvalues remain unchanged (default)
# @param discardImag[boolean] If True only the real part of the eigenvalues is kept (default: False)
# @return v[np.array] (m,m)-Matrix holding the eigenvectors (in the columns)
# @return w[np.array] (m,)-Vector holding the eigenvalues
def eig(A, sort=None, atype=None, discardImag=False):
    [w,v] = seig(A)
    if atype != None:
        if atype == 'normed':
            w=np.absolute(w)
        else:
            raise ValueError('Type neither None nor normed')
    elif discardImag == True:
        w = w.real
    elif discardImag == False:
        pass
    else:
        #If nothing is defined, assume only real parts
        assert np.all(np.fabs(w.imag) <= 1e-9), 'Encountered unexpected complex values'
        w=w.real
    ind = np.argsort(w)
        
    if sort != None:
        if sort == 'ascending':
            r=range(len(ind))
        elif sort == 'descending':
            r=range(len(ind)-1, -1,-1)
        else:
            raise ValueError('sort needs to be either ascending, descending')
        w = w[r]
        v=v[:,r]
    
    return [v,w]

## Calculates the transformation matrix between the n-dimensional unit sphere and the n-dimensional ellipsoid
#
# The ellipsoid is defined by the matrix P (symmetric, positive definite) and scaled by the value alpha. The calculated transformation 
# matrix T_p maps any point on the unit sphere x_u onto a point on the surface of the ellipsoid x_p defined by P
# by x_p = T_p.x_u
# @param P[np.array]  symmetric, positive definite nxn matrix defining the ellipsoid
# @param alpha[float] Scaling factor for the size of the ellipsoid defaulted to 1.0
# @returns [np.array] nxn transformation matrix
def getT(P, alpha = 1.0):
    [E,V] = eig(P, discardImag=True)
    #Only take real parts (Imaginary should be zero anyways)
    return  dot(E, diag(1.0/sqrt(V)))*sqrt(alpha)
## Convenience functions to perform multiple matrix multiplications
#
# ndot(A,B,C) is equivalent to np.dot(A,np.dot(B,C))
# @param *args[np.arrays] At least two compatible matrices as np.array
# @return M[np.array] Matrix resulting from multiplication
def ndot(*args):
    M = dp(args[0])
    for k in range(1,len(args)):
        M=dot(M,args[k])
    return M

## Implements an augmented dictionary holding all the transformations of a subfunnel ( instance of @ref subFun )
#
# The dictionaries use the start and end time of the transition as keys. 
# !Attention: All times used within or passed to transitionDict will be rounded in order to avoid numerical problems!
# @page transitionDict 
class transitionDict(object):
    def __init__(self):
        ## startDict uses the the start time of the transition as key. The corresponding item is a list of all transitions (as @ref transStruct ) being accessible from that time on.
        self.startDict = {}
        ## stopDict  uses the the end time of the transition as key. The corresponding item is a list of all transitions (as @ref transStruct )being accessible up to this time point.
        self.stopDict = {}
        self.startKeys=[]
        self.stopKeys=[]
        self._roundN=globalRoundVal
    
    ## roundN can only be set during initialization and not be changed afterwards
    @property
    def roundN(self):
        return self._roundN
    
    ## Make it usable like a dictionary by using the startDict
    # @param key[float] Start time of desired transitions
    # @return [list] List of transStructs. All transitions start at the time 'key'
    def __getitem__(self, key):
        key = self.round(key, 'c') #starting keys are always 'ceils'
        return self.startDict[key]
    
    ## Add a new transition to the dictionary 
    # @param tSpanKey[tuple of floats] Tuple holding (start time, end time) as float
    # @param transition Transition (instance of @ref transStruct ) to be added
    def __setitem__(self, tSpanKey, transition):
        assert isinstance(transition, transStruct), 'Only transition being an instance of transStruct can be added to the dictionary.'
        
        #Round the tSpan
        if tSpanKey[0] in self.startKeys:
            self.startDict[tSpanKey[0]].append(transition)
        else:
            self.startDict[tSpanKey[0]] = [transition]
            #self.startKeys = sorted(self.startKeys.append(tSpanKey[0])) #TBD check why i have to split this up
            self.startKeys.append(tSpanKey[0])
            self.startKeys = sorted(self.startKeys)
        if tSpanKey[1] in self.stopKeys:
            self.stopDict[tSpanKey[1]].append(transition)
        else:
            self.stopDict[tSpanKey[1]] = [transition]
            #self.stopKeys = sorted(self.stopKeys.append(tSpanKey[1]))
            self.stopKeys.append(tSpanKey[1])
            self.stopKeys = sorted(self.stopKeys)
        return 0
    
    ## Different add style
    # @param transition [@ref transStruct] Transition to be added
    def add(self, transition):
        assert isinstance(transition, transStruct), 'Only transition being an instance of transStruct can be added to the dictionary.'
        self[transition.tSpan] = transition
        return 0

    ## Return the keys of the startDict in ascending order
    # @return [list of floats] Ascending list of all start times
    def keys(self):
        return self.startKeys

    ## Get all transitions starting between start and end
    # @page transitionDict.startIn
    # @param start [rounded float] Minimum start time of the transitions (as rounded float so pi (with standard values) for example is 3141593.0 for instance)
    # @param end [rounded float] Maximum start time of the transitions (as rounded float so pi (with standard values) for example is 3141593.0 for instance)
    # @return [list of transStructs] All transitions starting between start and end
    def startIn(self, intimes = [-np.Inf, np.Inf]):
        #Get all possible transitions starting in a certain time interval
        start = intimes[0]
        end = intimes[1]
        npkeys = np.array(self.startKeys)
        trueInd = np.flatnonzero(np.logical_and( npkeys >=start, npkeys <= end ))
        allTrans = []
        for anind in trueInd:
            for aTrans in self.startDict[self.startKeys[anind]]:
                allTrans.append( aTrans )
        
        return allTrans
    ## Like @ref transitionDict.startIn but taking the real continuous time as input
    def startIn2(self, intimes = [-np.Inf, np.Inf]):
        start = intimes[0]
        end = intimes[1]
        start = ground(start) #To nearest
        end = ground(end) #To nearest
        return self.startIn([start, end])
        
    ## Get all transitions ending between start and end
    # @page transitionDict.stopIn
    # @param start [rounded float] Minimum end time of the transitions (as rounded float so pi (with standard values) for example is 3141593.0 for instance)
    # @param end [rounded float] Maximum end time of the transitions (as rounded float so pi (with standard values) for example is 3141593.0 for instance)
    # @return [list of transStructs] All transitions ending between start and end
    def stopIn(self, intimes = [-np.Inf, np.Inf]):
        #Get all possible transitions in a certain time interval
        start = intimes[0]
        end = intimes[1]
        npkeys = np.array(self.stopKeys)
        trueInd = np.where(np.logical_and( npkeys >=start, npkeys <= end ), True)
        allTrans = []
        for anind in trueInd:
            for aTrans in self.startDict[ self.stopKeys[anind] ]:
                allTrans.append(aTrans)
        return allTrans
    
    ## Like @ref transitionDict.stopIn but taking the real continuous time as input
    def stopIn2(self, intimes = [-np.Inf, np.Inf]):
        start = intimes[0]
        end = intimes[1]
        start = ground(start) #To nearest
        end = ground(end) #To nearest
        return self.stopIn([start, end])
    
    ## Get all transitions bounded by start and end.
    #
    #Bounded by means that the start time of the transition has to be bigger than start and the end time of the transition has to be smaller than end.
    # @page transitionDict.boundedBy
    # @param start [rounded float]  Minimum start time of the transitions (as rounded float so pi (with standard values) for example is 3141593.0 for instance)
    # @param end [rounded float]  Maximum end time of the transitions (as rounded float so pi (with standard values) for example is 3141593.0 for instance)
    # @return [list of transStructs] All transitions bounded by start and end
    def boundedBy(self, intimes = [-np.Inf, np.Inf]):
        #Get all transitions that start and stop within the given limits
        start = intimes[0]
        end = intimes[1]
        allTrans = []
        for aTrans in self.startIn([start, end]):
            if aTrans.tSpan[1] <=end:
                allTrans.append(aTrans)
        return allTrans
    
    ## Like @ref transitionDict.boundedBy but taking the real continuous time as input
    def boundedBy2(self, intimes = [-np.Inf, np.Inf]):
        start = intimes[0]
        end = intimes[1]
        start = ground(start) #To nearest
        end = ground(end) #To nearest
        return self.boundedBy([start, end])
    
    
    ## Get all transitions available at a certain point in time
    # @page transitionDict.availableAt
    # @param tTrans[rounded float] Time where the returned transitions can possibly be taken (as rounded float so pi (with standard values) for example is 3141593.0 for instance)
    # @return [list of transStructs] All transitions available at tTrans
    def availableAt(self, tTrans):
        #Get all the transitions possible at a certain time
        possTrans = self.startIn([-np.Inf, tTrans])
        allTrans = []
        for aTrans in possTrans:
            if aTrans.tSpan[1] >= tTrans:#Check if available
                allTrans.append(aTrans)
        return allTrans
    ## Like @ref transitionDict.availableAt but taking the real continuous time as input
    def availableAt2(self, tTrans):
        tTrans = ground(tTrans) #To nearest
        return self.boundedBy(tTrans)
    
    ## Get all transitions available at a certain point in time to another subfunnel (instance of @ref subFun) specified by his ID
    # @page transitionDict.availableAtTo
    # @param clockSet [tuple of rounded floats] clockSet[0]:Time where the returned transitions can possibly be taken; clockSet[1]:C2_p clock (as rounded float so pi (with standard values) for example is 3141593.0 for instance)
    # @param cFunnelID [string] Unique subfunnel ID corresponding to the desired target subfunnel
    # @param arriveTime [rounded float] Value of C1_c in the transition set. If given than it has to hold that C1_p==arriveTime(after the transition)
    # @return [list of transStructs] All transitions available at tTrans into the subfunnel cFunnelID
    def availableAtTo(self, C1_p, cFunnelID, arriveTime=None, C2_p = None):
        #Get all possible transitions to a specific funnel at a certain time
        assert (arriveTime==None or (arriveTime!=None and C2_p!=None)), 'If arriveTime is given, than the second clock has to be given too!'
        clockSet=[C1_p, C2_p]
        possTrans = self.availableAt(clockSet[0])
        allTrans = []
        for aTrans in possTrans:
            check = False
            if aTrans.childID == cFunnelID:
                if arriveTime==None:
                    check = True
                else:
                    _unused, [C1_c, _unused] = aTrans.doTrans(clockSet[0], clockSet[1]) #Attention doTrans returns rounded float values for the clocks (3141593.0 instead of pi)
                    if C1_c == arriveTime:
                        check=True
                if check:
                    allTrans.append(aTrans)
        return allTrans
    ## Like @ref transitionDict.availableAtTo but taking the real continuous time as input
    def availableAtTo2(self, C1_p, cFunnelID, arriveTime=None, C2_p = None):
        C1_p = ground(C1_p) #To nearest
        C2_p = ground(C2_p) #To nearest
        return self.availableAtTo(C1_p, cFunnelID, arriveTime=None, C2_p = None)
    
## Helper function to circumvent the usage of the original python exec
def myExec(tS, C1_p, C2_p):
    tS=''.join(tS.split())
    tSL=tS.split(';')#Different assignements have to be semicolon seperated
    C1_c=C2_c=None
    for atS in tSL:
        [vname, vvalue] = atS.split('=')#Assignments are done via normal =
        #vars()[vname]=float(eval(vvalue))#No clue why this does not work TBD find sth elegant
        if vname=='C1_c': C1_c = float(eval(vvalue))
        if vname=='C2_c': C2_c = float(eval(vvalue))
    assert (C1_c!=None and C2_c!=None), 'Assignment failed for: '+tS
    return [C1_c, C2_c]

## Class to stock informations of evaluable strings
class myEvalString(object):
    def __init__(self, specListIn, evalStrIn='{0:d}', plotStrIn='{0:.3f}'):
        assert len(specListIn)==3, '''Error specList has to be of the form ['>;>=;==;=<;<;=', float;str, float;str] where ';' means 'or'  ''' 
        self.specList = dp(specListIn)

        # fix for tchecker
        # Clocks always have to be on the left
        if isinstance(self.specList[2], str) and (self.specList[0] in ['<=','<','==','>','>=']):
            # Reverse
            compareChangeDict = {'<=':'>=','<':'>','==':'==','>':'<', '>=':'<='}
            self.specList = [compareChangeDict[self.specList[0]], self.specList[2], self.specList[1]]
        self.specList[0] = '{0}' + self.specList[0] + '{1}' 
        self.evalStr = evalStrIn
        self.plotStr = plotStrIn

        self.mode = "tchecker"
    
    def getStr(self, atype = 'eval'):
        expr = dp(self.specList[0])
        val1 = dp(self.specList[1])
        val2 = dp(self.specList[2])

        # Tchecker does not support clock assignements of the type x=x
        if val1 == val2:
            return ""

        fexpr = self.evalStr if atype == 'eval' else self.plotStr
        
        if isinstance(val1, str):
            expr = expr.format(val1, '{0}')
        else:
            if atype=='plot': 
                val1 = float(guround(val1))
            else:
                val1 = int(val1)
            expr = expr.format(fexpr.format(val1), '{0}')
        if isinstance(val2, str):
            expr = expr.format(val2)
        else:
            if atype=='plot': 
                val2 = float(guround(val2))
            else:
                val2 = int(val2)
            expr = expr.format(fexpr.format(val2))
        return expr
        

## Class to conveniently store transitions between subfunnels
#
# Each subfunnel ( @ref subFun ) has in general two associated clocks called C1 and C2. These are used to specify the transition guards and assignments by the following notation:
# The clocks of the parent funnel (from funnel) are called C1_p and C2_p, the clocks of the child funnel (to funnel) are called C1_c and C2_c.
# @page transStruct 
class transStruct(object):
#     ## Constructor
#     # @param parentIDIn[string] Unique ID qualifying the 'from funnel'. Should be the same as the ID of subfunnel ( @ref subFun ) holding the @ref transitionDict in which the transStruct is stored
#     # @param childIDIn[string] Unique ID qualifying the 'to funnel'.
#     # @param transGuardIn[string] Evaluable string specifying the transition guard like "a<=C1_p<=b [and C2_p [=>>=<<=] d]" with a,b and d being constants 
#     # @param transSetIn[string] Assignments to the clock set in the child funnel of the form "C1_c=c; C2_c=0.0" with d being a constant
#     # @param tSpan[tuple of floats] Characterizing the time (in terms of C1) where the transition is possibly accessible. In general tSpan=(a,b) (see transGuard) 
#     # @param transGuardPLOTIn[string] Short form of the transition guard used in plots for example. Defaulted to transGuardIn
#     # @param transSetPLOTIn[string] Short form of the transition assignment used in plots for example. Defaulted to transSetIn
#     def __init__(self, parentIDIn, childIDIn, transGuardIn, transSetIn, tSpan, transGuardPLOTIn=None, transSetPLOTIn=None):
#         self.parentID = parentIDIn
#         self.childID = childIDIn
#         self.transGuard = transGuardIn
#         self.transGuardPLOT = transGuardPLOTIn if transGuardPLOTIn!=None else transGuardIn 
#         #transition guards must be in the form
#         #tg : a<=C1_p<=b (and C2_p [=>>=<<=] d)
#         self.transSet = transSetIn
#         self.transSetPLOT = transSetPLOTIn if transSetPLOTIn!=None else transSetIn
#         #Transition sets must be of the form
#         #ts : C1_c = (c or alpha*C1_p); C2_c=0;
#         self.tSpan = tSpan #Holds (a,b)
    def __init__(self, parentIDIn, childIDIn, tSpanIn, transGuardListIn, transSetListIn, evalStrIn='{0:d}', plotStrIn='{0:.3f}'):
        self.parentID = parentIDIn
        self.childID = childIDIn
        self.tSpan = tSpanIn #rounded float
        self.transGuardList = [] #rounded float
        for atrans in transGuardListIn:
            self.transGuardList.append( myEvalString(atrans, evalStrIn, plotStrIn) )
        self.transSetList = [] #rounded float
        for aset in transSetListIn:
            self.transSetList.append( myEvalString(aset, evalStrIn, plotStrIn) )
        
        self.transGuard = None
        self.transGuardPLOT = None
        self.transSet = None
        self.transSetPLOT = None
        #Assemble
        self.roundN = globalRoundVal

        # New tchecker variable
        self.event = "noaction"
    
    @property 
    def roundN(self):
        return self._roundN
    @roundN.setter
    def roundN(self, roundNIn):
        self._roundN = roundNIn
        self.transGuard = None
        self.transGuardPLOT = None
        for atrans in self.transGuardList:
            pltExpr = atrans.getStr(atype = 'plot')
            evalExpr = atrans.getStr(atype = 'eval')
            self.transGuardPLOT = self.transGuardPLOT+' and '+pltExpr if self.transGuardPLOT!=None else pltExpr
            self.transGuard = self.transGuard+' and '+evalExpr if self.transGuard!=None else evalExpr
        self.transSet = None
        self.transSetPLOT = None
        for aset in self.transSetList:
            pltExpr = aset.getStr(atype = 'plot')
            evalExpr = aset.getStr(atype = 'eval')
            self.transSetPLOT = self.transSetPLOT+'; '+pltExpr if self.transSetPLOT!=None else pltExpr
            self.transSet = self.transSet+'; '+evalExpr if self.transSet!=None else evalExpr
        return 0
    
    ## Function used to 'simulate' a transition between two subfunnels ( @ref subFun ).
    #
    # Checks the transGuards and invariants of the parent and child funnel (if given) and returns the success status and the new clock set
    # @param C1_p[rounded float] Current C1 clock of the parent funnel (as rounded float so pi (with standard values) for example is 3141593.0 for instance)
    # @param C2_p[rounded float] Current C2 clock of the parent funnel (as rounded float so pi (with standard values) for example is 3141593.0 for instance)
    # @param pFunnel[ @ref subFun ] parent funnel ('from funnel'). If not given invariants on funnel remain unchecked
    # @param cFunnel[ @ref subFun ] child funnel ('to funnel'). If not given invariants on funnel remain unchecked
    # @param Xglob[np.array] Current state of the dynamical system (point in the state space) in global coordinates. If given than it is checked wether the point lies inside both funnels
    # @return [list] list[0] is the success status (as boolean) and list[1] is a tuple of floats representing [C1_c and C2_c] after the transition
    def doTrans(self, C1_p, C2_p, pFunnel=None, cFunnel=None, Xglob=None):
        #Check if success:
        valid = eval(self.transGuard)
        
        #This does not work as desired 
        #exec(self.transSet) #This creates C1_c and C2_c
        #Trying
        C1_c, C2_c = myExec(self.transSet, C1_p, C2_p)
        
        #Also check of invariants are respected if the funnels are given
        if pFunnel!=None and cFunnel!=None:
            valid = valid and eval(pFunnel.invariantList) 
            valid = valid and eval(cFunnel.invariantList)
            if Xglob != None:
                valid = valid and pFunnel.globPinF(Xglob, guround(C1_p) )#Check if the point is in the ellipse representing the from funnel at time t=C1_p
                valid = valid and cFunnel.globPinF(Xglob, guround(C1_c) ) #Check if the point is in the ellipse representing the to funnel at time t=C1_c 
        
        return [valid, [C1_c, C2_c]]#Return whether it succeeded and the clocks for the child funnel
        
        

## Helper functions to change between the different calculations types for the intersections
def getPossibleTrans(funF, funT, dLfIn=0.001, tScale=1. ):
    #return getPossibleTrans_V1(funF, funT)
    #return getPossibleTrans_vC(funF, funT, dLf=dLfIn)
    return getPossibleTrans_vC_VecVers(funF, funT, dLf=dLfIn, tScale=tScale)


## Sample based search algorithm - Vectorized version for speed-up
#
# This algorithm computes the transitions between two subfunnels ( @ref subFun ) distributes along the nominal trajectory based on the van der corput sequence ( @ref vdc ).
# @param funF[ @ref subFun ] Parent funnel (from funnel)
# @param funT[ @ref subFun ] Child funnel (to funnel)
# @param dLf[float] Trajectory position delta used to finely search the limits of a possible transition once a starting point was found. Defaulted to 0.001 (equals 1000 points along the nominal trajectory)
# @return allTransFT[list of @ref transStruct] List of all possible transitions found from the parent to the child funnel
# @page getPossibleTrans_vC
# @todo Check automatically if globalRoundN is big enough
# @todo Improve performance by using the derivative for searching the actual transition limits
# @todo removable double evaluation

def getPossibleTrans_vC_VecVers(funF, funT, dLf=0.001, tScale=1. ):
    up = lambda x: int(np.ceil(x * tScale))
    down = lambda x: int(np.floor(x * tScale))
    mid = lambda x: int(x*tScale)

    #Searching the to funnel with a van der corput pattern
    #dLf: When a principally possible transition is found, use dLf to search the actual boundaries
    m = funF.dynSys.stateDim
    #For the moment just search a certain number and do not limit time
    #lambdaSearchT =  np.array(globVDC.getN(funT.traj.N_searchTo)).squeeze()
    #lambdaSearchT =  np.array(globVDC.getN(funT.traj.N_searchTo)).squeeze()
    if funT.traj.isCycFlag:
        lambdaSearchT =  np.linspace( 0.0, 1.0, funT.traj.N_searchTo+1 )[0:-1]
    else:
        lambdaSearchT =  np.linspace( 0.0, 1.0, funT.traj.N_searchTo )
    if funF.traj.isCycFlag:
        lambdaSearchF =  np.linspace( 0.0, 1.0, funF.traj.N_searchFrom+1 )[0:-1]
    else:
        lambdaSearchF =  np.linspace( 0.0, 1.0, funF.traj.N_searchFrom+1 )
    #Translate it into valid times
    tT = funT.traj.unitToTime(lambdaSearchT)
    tF = funF.traj.unitToTime(lambdaSearchF)
    
    #Only search on rounded points
    tT = gToNearest(tT)
    tF = gToNearest(tF)
    
    
    allXT = funT.traj.getX(tT)
    allXF = funF.traj.getX(tF)
    
    lambdaArriveList = []
    tSpanList = []
    allTransFT = []
    #Calculate some necessary values
    transF = getT(funF.dynSys.getP(), funF.lyapVal)
    transTui = inv(getT(funT.dynSys.getP()))
    [_unused, sv, _unused] = svd(dot(transTui, transF))
    lVadd = namax(sv)#TBD:check with different QR conservative minimum additional value to cover to-funnel ellipse 
    
    rootlVT = funT.lyapVal**0.5
    
    #Preliminary check: If the ellipse of the fromFunnel does not fit into the ellipse of the ToFunnel -> no need to search (adding small coefficient
    #To ensure a minimum time interval)
    if lVadd*1.05 >= rootlVT:
        return [] 
    
    #Initialize to make them accesible by nested functions
    dX = np.zeros(allXF.shape)
    inRange = np.zeros(dX.shape[1], dtype=np.bool)
    
    #Define a nested functions
    def checkDX(aDX):
        aDX_T = dot(transTui, aDX.reshape((m,aDX.size//m)));
        #The fromEllipse around aDX is in the toEllipse around zero
        # From http://stackoverflow.com/q/19094441/166749
        return ( rootlVT > lVadd + np.sqrt(np.einsum('ij,ij->j', aDX_T, aDX_T)) )
    
    #Get a certain transition
    def getThisTrans(k,l):
        thistT = tT[k]
        thisTlower = max(thistT-0.5/globalRoundVal, funT.tSpan[0])
        thisTupper = min(thistT+0.5/globalRoundVal, funT.tSpan[1])
        
        thisXTlower = funT.traj.getX(thisTlower)
        thisXTupper = funT.traj.getX(thisTupper)
        
        #Getting true start time in from Funnel
        thisLambdaF = lambdaSearchF[l]
        thisLambdaFf = max(0.0, thisLambdaF-dLf)
        #while checkDX( funF.traj.getX(funF.traj.unitToTime(thisLambdaF)) - thisXT ) and ((thisLambdaF-dLf)>0 or funF.traj.isCycFlag) and thisLambdaF>-1.0:
        #Perform evaluation at rounded times
        check=True
        doCont = True
        while doCont:
            nextT = gToNearest(funF.traj.unitToTime(thisLambdaFf))
            if nextT < funF.tSpan[0]:
                #Rounding lead to smaller value
                break
            nextX = funF.traj.getX(nextT)
            doCont = doCont and checkDX( nextX - thisXTlower ) and checkDX( nextX - thisXTupper ) 
            doCont = doCont and thisLambdaFf>=0 and check
            if not doCont:
                break
            if thisLambdaFf == 0.0:
                check=False
            thisLambdaF = thisLambdaFf
            thisLambdaFf = max(0.0, thisLambdaFf-dLf)
            #Done
        startLambda = thisLambdaF
        
        #Get the last valid large step lambda
        roughLout = l
        for thisK in range(l, tF.size-1):
            roughLout = thisK
            if not inRange[thisK+1]:
                break
              
             
        thisLambdaF = lambdaSearchF[roughLout]
        thisLambdaFf = min(1.0, thisLambdaF+dLf)
        #while checkDX(funF.traj.getX( funF.traj.unitToTime(thisLambdaF)) - thisXT ) and ((thisLambdaF+dLf)<1.0 or funF.traj.isCycFlag) and thisLambdaF<2.0:
        
        check=True
        doCont = True
        while doCont:
            nextT = gToNearest(funF.traj.unitToTime(thisLambdaFf))
            if nextT > funF.tSpan[1]:
                #Rounding lead to smaller value
                break
            nextX = funF.traj.getX(nextT)
            doCont = doCont and checkDX( nextX - thisXTlower ) and checkDX( nextX - thisXTupper ) 
            doCont = doCont and thisLambdaFf>=0 and check
            if not doCont:
                break
            if thisLambdaFf == 1.0:
                check=False
            thisLambdaF = thisLambdaFf
            thisLambdaFf = min(1.0, thisLambdaFf+dLf)
        stopLambda = thisLambdaF
        #Calculate where to continue
        if thisLambdaF+dLf >= 1.0:
            lout = tF.size
        else:
            lout = ffind( lambdaSearchF > thisLambdaF+dLf )
        while lout <= l:
            lout+=1
        
        lambdaArriveList.append( lambdaSearchT[k] )
        tSpanList.append([startLambda, stopLambda])
        
        #this should never be necessary now; Will be removed soon
        if startLambda == 0 and 0.0<=stopLambda and stopLambda<1.0:
            aTcouple = [float(funF.traj.tSpan[0]), float(funF.traj.unitToTime(stopLambda))]
        elif 0.0<startLambda and startLambda<=1.0 and stopLambda==1.0:
            aTcouple = [float(funF.traj.unitToTime(startLambda)), float(funF.traj.tSpan[1])]
        elif startLambda == 0 and stopLambda==1.0:
            aTcouple = [float(funF.traj.tSpan[0]), float(funF.traj.tSpan[1])]
        else:
            aTcouple = [float(funF.traj.unitToTime(startLambda)), float(funF.traj.unitToTime(stopLambda))]
        #Add
        #allTransFT.append(transStruct(funF.ID, funT.ID, '{0:.12e}<=C1_p and C1_p<={1:.12e}'.format(aTcouple[0], aTcouple[1]), 'C1_c={0:.12e};C2_c=0.0'.format(thistT), aTcouple, transGuardPLOTIn='{0:.4f}<=C1_p and C1_p<={1:.4f}'.format(aTcouple[0], aTcouple[1]), transSetPLOTIn='C1_c={0:.4e};C2_c=0.0'.format(thistT)))
        #Get the rounded float values ensure consistency and add
        thistT = ground(thistT)
        aTcouple = ground(aTcouple)
        if aTcouple[1]-aTcouple[0] > 2.0:
            #Take the next rounding point, otherwise it could happan that ellipsoids overlap
            #If this is not possible than the transition might not be save
            aTcouple[0] += 1.0
            aTcouple[1] -= 1.0
            allTransFT.append(transStruct(funF.ID, funT.ID, aTcouple, [['<=', down(aTcouple[0]), 'C1_p'], ['<=', 'C1_p', up(aTcouple[1]),]], [['=', 'C1_c', mid(thistT)], ['=', 'C2_c', 0]] ) )
        
        getAdditionalTrans()
        return lout
        #####
    
    def getAdditionalTrans():
        #Check if it makes sense to add additional arriving points
        #TBD
        pass
        
        
    #Loop over XT to find rough intersections
    for k in range(tT.size):
        inRange = False
        l = 0;
        #Vectorize the caculations of one to point
        dX = allXF - np.reshape(allXT[:,k],(m,1)) #Equivalent allXF - repmat(allXT,1,size(allXF,2)) in Matlab
        inRange = checkDX(dX)
        while l < tF.size:
            if inRange[l]:
                l = getThisTrans(k,l)
            else:
                l += 1
    
    return allTransFT



## Sample based search algorithm
#
# This algorithm computes the transitions between two subfunnels ( @ref subFun ) distributes along the nominal trajectory based on the van der corput sequence ( @ref vdc ).
# @param funF[ @ref subFun ] Parent funnel (from funnel)
# @param funT[ @ref subFun ] Child funnel (to funnel)
# @param dLf[float] Trajectory position delta used to finely search the limits of a possible transition once a starting point was found. Defaulted to 0.001 (equals 1000 points along the nominal trajectory)
# @return allTransFT[list of @ref transStruct] List of all possible transitions found from the parent to the child funnel
# @page getPossibleTrans_vC
# @todo Check automatically if globalRoundN is big enough
# @todo Improve performance by using the derivative for searching the actual transition limits
# @todo removable double evaluation
def getPossibleTrans_vC(funF, funT, dLf=0.001 ):
    #Searching the to funnel with a van der corput pattern
    #dLf: When a principally possible transition is found, use dLf to search the actual boundaries
    m = funF.dynSys.stateDim
    #For the moment just search a certain number and do not limit time
    #lambdaSearchT =  np.array(globVDC.getN(funT.traj.N_searchTo)).squeeze()
    #lambdaSearchT =  np.array(globVDC.getN(funT.traj.N_searchTo)).squeeze()
    if funT.traj.isCycFlag:
        lambdaSearchT =  np.linspace( 0.0, 1.0, funT.traj.N_searchTo+1 )[0:-1]
    else:
        lambdaSearchT =  np.linspace( 0.0, 1.0, funT.traj.N_searchTo )
    if funF.traj.isCycFlag:
        lambdaSearchF =  np.linspace( 0.0, 1.0, funF.traj.N_searchFrom+1 )[0:-1]
    else:
        lambdaSearchF =  np.linspace( 0.0, 1.0, funF.traj.N_searchFrom+1 )
    #Translate it into valid times
    tT = funT.traj.unitToTime(lambdaSearchT)
    tF = funF.traj.unitToTime(lambdaSearchF)
    
    #Only search on rounded points
    tT = gToNearest(tT)
    tF = gToNearest(tF)
    
    
    allXT = funT.traj.getX(tT)
    allXF = funF.traj.getX(tF)
    
    lambdaArriveList = []
    tSpanList = []
    allTransFT = []
    #Calculate some necessary values
    transF = getT(funF.dynSys.getP(), funF.lyapVal)
    transTui = inv(getT(funT.dynSys.getP()))
    [_unused, sv, _unused] = svd(dot(transTui, transF))
    lVadd = namax(sv)#TBD:check with different QR conservative minimum additional value to cover to-funnel ellipse 
    
    rootlVT = funT.lyapVal**0.5
    
    #Preliminary check: If the ellipse of the fromFunnel does not fit into the ellipse of the ToFunnel -> no need to search (adding small coefficient
    #To ensure a minimum time interval)
    if lVadd*1.05 >= rootlVT:
        return [] 
    
    #Define a nested functions
    def checkDX(aDX):
        return ( rootlVT > lVadd + norm(dot(transTui, aDX.reshape((m,1)))) )#The fromEllipse around aDX is in the toEllipse around zero#TBD replace with cholesky and sparse matrix?
    
    #Get a certain transition
    def getThisTrans(k,l):
        thistT = tT[k]
        thisTlower = max(thistT-0.5/globalRoundVal, funT.tSpan[0])
        thisTupper = min(thistT+0.5/globalRoundVal, funT.tSpan[1])
        
        thisXTlower = funT.traj.getX(thisTlower)
        thisXTupper = funT.traj.getX(thisTupper)
        
        #Getting true start time in from Funnel
        thisLambdaF = lambdaSearchF[l]
        thisLambdaFf = max(0.0, thisLambdaF-dLf)
        #while checkDX( funF.traj.getX(funF.traj.unitToTime(thisLambdaF)) - thisXT ) and ((thisLambdaF-dLf)>0 or funF.traj.isCycFlag) and thisLambdaF>-1.0:
        #Perform evaluation at rounded times
        check=True
        doCont = True
        while doCont:
            nextT = gToNearest(funF.traj.unitToTime(thisLambdaFf))
            if nextT < funF.tSpan[0]:
                #Rounding lead to smaller value
                break
            nextX = funF.traj.getX(nextT)
            doCont = doCont and checkDX( nextX - thisXTlower ) and checkDX( nextX - thisXTupper ) 
            doCont = doCont and thisLambdaFf>=0 and check
            if not doCont:
                break
            if thisLambdaFf == 0.0:
                check=False
            thisLambdaF = thisLambdaFf
            thisLambdaFf = max(0.0, thisLambdaFf-dLf)
            #Done
        startLambda = thisLambdaF
        thisLambdaF = lambdaSearchF[l]
        thisLambdaFf = min(1.0, thisLambdaF+dLf)
        #while checkDX(funF.traj.getX( funF.traj.unitToTime(thisLambdaF)) - thisXT ) and ((thisLambdaF+dLf)<1.0 or funF.traj.isCycFlag) and thisLambdaF<2.0:
        
        check=True
        doCont = True
        while doCont:
            nextT = gToNearest(funF.traj.unitToTime(thisLambdaFf))
            if nextT > funF.tSpan[1]:
                #Rounding lead to smaller value
                break
            nextX = funF.traj.getX(nextT)
            doCont = doCont and checkDX( nextX - thisXTlower ) and checkDX( nextX - thisXTupper ) 
            doCont = doCont and thisLambdaFf>=0 and check
            if not doCont:
                break
            if thisLambdaFf == 1.0:
                check=False
            thisLambdaF = thisLambdaFf
            thisLambdaFf = min(1.0, thisLambdaFf+dLf)
        stopLambda = thisLambdaF
        #Calculate where to continue
        if thisLambdaF+dLf >= 1.0:
            lout = tF.size
        else:
            lout = ffind( lambdaSearchF > thisLambdaF+dLf )
        while lout <= l:
            lout+=1
        
        lambdaArriveList.append( lambdaSearchT[k] )
        tSpanList.append([startLambda, stopLambda])
        
        #this should never be necessary now; Will be removed soon
        if startLambda == 0 and 0.0<=stopLambda and stopLambda<1.0:
            aTcouple = [float(funF.traj.tSpan[0]), float(funF.traj.unitToTime(stopLambda))]
        elif 0.0<startLambda and startLambda<=1.0 and stopLambda==1.0:
            aTcouple = [float(funF.traj.unitToTime(startLambda)), float(funF.traj.tSpan[1])]
        elif startLambda == 0 and stopLambda==1.0:
            aTcouple = [float(funF.traj.tSpan[0]), float(funF.traj.tSpan[1])]
        else:
            aTcouple = [float(funF.traj.unitToTime(startLambda)), float(funF.traj.unitToTime(stopLambda))]
        #Add
        #allTransFT.append(transStruct(funF.ID, funT.ID, '{0:.12e}<=C1_p and C1_p<={1:.12e}'.format(aTcouple[0], aTcouple[1]), 'C1_c={0:.12e};C2_c=0.0'.format(thistT), aTcouple, transGuardPLOTIn='{0:.4f}<=C1_p and C1_p<={1:.4f}'.format(aTcouple[0], aTcouple[1]), transSetPLOTIn='C1_c={0:.4e};C2_c=0.0'.format(thistT)))
        #Get the rounded float values ensure consistency and add
        thistT = ground(thistT)
        aTcouple = ground(aTcouple)
        if aTcouple[1]-aTcouple[0] > 2.0:
            #Take the next rounding point, otherwise it could happan that ellipsoids overlap
            #If this is not possible than the transition might not be save
            aTcouple[0] += 1.0
            aTcouple[1] -= 1.0
            allTransFT.append(transStruct(funF.ID, funT.ID, aTcouple, [['<=', aTcouple[0], 'C1_p'], ['<=', 'C1_p', aTcouple[1],]], [['=', 'C1_c', thistT], ['=', 'C2_c', 0.0]] ) )
        
        getAdditionalTrans()
        return lout
        #####
    
    def getAdditionalTrans():
        #Check if it makes sense to add additional arriving points
        #TBD
        pass
        
        
    #Loop over XT to find rough intersections
    for k in range(tT.size):
        inRange = False
        l = 0;
        while l < tF.size:
            #Check if this pair is in range
            dXkl = allXF[:,l] - allXT[:,k]
            inRange = checkDX(dXkl)
            if inRange:
                l = getThisTrans(k,l)
            else:
                l += 1
    
    return allTransFT
        
    
    
## Old search algorithm
def getPossibleTrans_V1(funF, funT):
    ###Calculates all possible transitions from funF to funT
    #dimension
    m = funF.dynSys.stateDim
    #TBD Sampling based code could be largely improved by variable step size or other features
    #TBD check if analytic solutions for specific trajectories (linear...) are of interest
    #Get all he sampling points
    tF = np.arange(funF.traj.tSpan[0], funF.traj.tSpan[1], funF.traj.dT)
    allXF = funF.traj.getX(tF)
    tT = np.arange(funT.traj.tSpan[0], funT.traj.tSpan[1], funT.traj.dT)
    allXT = funT.traj.getX(tT)
    
    #Calculate some necessary values
    transF = getT(funF.dynSys.getP(), funF.lyapVal)
    transTui = inv(getT(funT.dynSys.getP()))
    [_unused, sv, _unused] = svd(dot(transTui, transF))
    lVadd = namax(sv)#TBD:recheck with different QR conservative minimum additional value to cover to-funnel ellipse 
    
    rootlVT = funT.lyapVal**0.5
    
    #Preliminary check: If the ellipse of the fromFunnel does not fit into the ellipse of the ToFunnel -> no need to search (adding small coefficient
    #To ensure a minimum time interval)
    if lVadd*1.05 >= rootlVT:
        return [] 
    
    #Define a nested functions
    def checkDX(aDX):
        return ( rootlVT > lVadd + norm(dot(transTui, aDX)) )#The fromEllipse around aDX is in the toEllipse around zero#TBD replace with cholesky and sparse matrix?
    
    def getThisTrans(KfromMid, TtoMid):
        ###calculate the transition form funF to funT for funT=funT(TtoMid)
        TXmid = funT.traj.getX(TtoMid)
        #inRangeN1 = False
        #for k in range(tF.size):
        #    dXk = allXF[:,k] - TXmid
        #    inRangeN2 = checkDX(dXk)
        #    if inRangeN1==False and inRangeN2==True:
        #        inRangeN1=True
        #        tStartF = tF[k]
        #    if inRangeN1==True and inRangeN2==False:
        #        tEndF = tF[k-1]
        #        break
        kn = KfromMid
        dXk = allXF[:,kn].reshape((m,1)) - TXmid
        while checkDX(dXk):
            kn-=1
            dXk = allXF[:,kn].reshape((m,1)) - TXmid
        tStartFn = tF[kn+1]
        
        kn = KfromMid
        dXk = allXF[:,kn].reshape((m,1)) - TXmid
        while checkDX(dXk):
            kn+=1
            dXk = allXF[:,kn].reshape((m,1)) - TXmid
        tEndFn = tF[kn-1]
            
        #Assemble the transition
        tStartFfn = min(tStartFn, tEndFn)
        tEndFfn = max(tStartFn, tEndFn)
        return transStruct(funF.ID, funT.ID, '{0:.12e}<=C1_p and C1_p<={1:.12e}'.format(tStartFfn, tEndFfn), 'C1_c={0:.12e};C2_c=0.0'.format(TtoMid), [tStartFfn, tEndFfn], transGuardPLOTIn='{0:.4f}<=C1_p and C1_p<={1:.4f}'.format(tStartFfn, tEndFfn), transSetPLOTIn='C1_c={0:.4e};C2_c=0.0'.format(TtoMid))
        
    
    inRange1 = False
    allTransFT = []
    #Adaption for cyclic traj
    for k in range(tF.size):
        for l in range(tT.size):
            #Check if this pair is inrange
            dXkl = allXF[:,k] - allXT[:,l]
            inRange2 = checkDX(dXkl)
            #Now compare the different ranges and draw conclusions
            if inRange2:
                kEndF = k
                tEndT = tT[l]
                break #There is a possible transition between funF and funT
        if inRange1==False and inRange2==True:
            #First point on fromFunnel where a change is possible
            inRange1=True
            kStartF = k
            tStartT = tT[l]
        if inRange1==True and inRange2==False:
            #first point where no transition is possible
            #One intersection interval is determined get the corresponding translation
            allTransFT.append( getThisTrans(np.rint(0.5*(kStartF+kEndF)), 0.5*(tStartT+tEndT)) )#Fix the arriving point
            inRange1=False
    return allTransFT
            
            
        
## Function to determine the maximal 'size' (as lyapVal) of innerEllip so that it is 
# a subset of outerEllip. The offset has to be defined in global coordinates
#
# @params outerEllip [np.array] positive definite square matrix definning the shape of the outer ellipse
# @params outerSize [float] Size of the outer ellipse
# @params innerEllip  [np.array] positive definite square matrix definning the shape of the inner ellipse
# @params offSet [np.array-column vector] Offset between the center of the ellipsoids in global coords
# @return [float] maximum size of the innerEllipse so that it is confined in the outerellipse

def getMaxInnerSize(outerEllip, outerSize, innerEllip, offset=None):
    
    Tou = getT(outerEllip)
    Toui = inv(Tou)
    Tiu = getT(innerEllip)
    
    if offset!=None:
        addVal = 0.0
    else:
        addVal = norm(dot(Toui, offset.reshape((offset.size,1))))
    
    [_unused, sv, _unused] = svd(dot(Toui, Tiu))
    lVinner = namax(sv)
    
    return ((outerSize**0.5-addVal)/lVinner)**2.0
    
    

    
