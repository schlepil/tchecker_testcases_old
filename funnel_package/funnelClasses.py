## @package funnelClasses
# Holds important classes for the construction of funnel based timed automata 

import string

import control as ctrl

import funnel_package.mechanicalSysPack as msp
from funnel_package.utils import *
import funnel_package.plotUtils as pu
from copy import deepcopy

## Class repreenting a subfunnel.
# 
# A subfunnel consists of a nominal trajectory (e.g. @ref analyticTraj) and a dynamical system (e.g. @ref dynSys).
# The dynamical system has to be 'complete' meaning already parametrized with QR for example
# @page subFun
class subFun(astruct):
    ## Constructor
    def __init__(self):
        super(subFun, self).__init__()
        ## Instance of @ref transitionDict holding all transition originating from this subfunnel
        self.transDict = transitionDict()
        ## List of strings (have to be evaluable) representing the set of invariants that have to be respected 
        self.invariantListExpr = []
        self.invariantList = None
        self.invariantListPLOT = None
        ## Nominal trajectory (e.g. @ref analyticTraj)
        self.traj = None
        ## Dynamical system that is supposed to follow the nominal trajectory (e.g. @ref dynSys).
        self.dynSys = None
        ## Reference to the property of the trajectory
        self._tSpan = None
        ## The discretised tSpan
        self._tSpanD = None
        ## Reference to the property of the trajectory
        self._alphaTime = None
        ## Convenience property that can be used to reshape arrays
        self._shape = None
        ## Fine search parameter
        self.dLf = None
    
    #Easy access to trajectory properties
    @property
    def tSpan(self):
        return self.traj.tSpan
    @tSpan.setter
    def tSpan(self):
        raise AssertionError('The span can not be set like this. It is creating when instantiating the trjectory class')
    @tSpan.deleter
    def tSpan(self):
        raise AssertionError('tSpan can not be deleted')
    @property
    def tSpanD(self):
        assert self.traj != None, 'No trajectory is yet assigned to the subfunnel!'
        self._tSpanD = ground(self.tSpan)
        return self._tSpanD
    @property
    def alphaTime(self):
        assert self.traj != None, 'No trajectory is yet assigned to the subfunnel!'
        return self.traj.alphaTime
    @alphaTime.setter
    def alphaTime(self, value):
        assert self.traj != None, 'No trajectory is yet assigned to the subfunnel!'
        self.traj.alphaTime = value
        return 0
    @alphaTime.deleter
    def alphaTime(self):
        raise AssertionError('alphaTime can not be deleted')
    @property
    def shape(self):
        #Returns the shape of the state vector
        assert self.dynSys != None, 'No dynamical system is yet assigned to the subfunnel!'
        return (self.dynSys.stateDim,1)
    
    ## Calculates the lyapunov value associated with a point given relative to the nominal trajectory
    # @param dX [_shape-np.array] Distance between a point in the state-space and its reference position on the trajectory
    # return [float] Lyapunov value of this point
    def locVofP(self, dX):
        #return norm(dot(self.dynSys.getPc(), dX))#There seems to be a problem
        return ndot(dX.T, self.dynSys.getP(), dX)
    
    ## Returns whether a point given in relative coordinates is inside or outside the funnel
    # @param dX [_shape-np.array] Distance between a point in the state-space and its reference position on the trajectory
    # return [boolean] Point in the funnel (the ellipse representing it) or not
    def locPinF(self, dX):
        return self.locVofP(dX)<=self.lyapVal
    
    ## Returns whether a point given in global coordinates is inside or outside the funnel
    # @param X [_shape-np.array] Point in the state-space in global coordinates
    # @param C1_p [float,None] Time specifying the center of the ellipse representing the funnel for a particular time point. If given it will be checked if X is in this ellipse. If None(default) the whole trajectory will be searched if there is any time point for which this point is in the funnel
    # return [float] Minimal lyapunov value of this point
    def globVofP(self, X, C1_p=None):
        #Calculates the minimal lyapVal for a given global point if no time is given, otherwise the value for this time and point is calculated
        if C1_p==None:
            t = self.traj.unitToTime( np.linspace(0.0, 1.0, self.N_searchTo) )
        else:
            t=np.array(C1_p)
                
        Xt=self.traj.getX(t)
        minV=np.Inf
        for k in range(t.size):
            minV = min( minV, self.locVofP( (X-Xt[:,k].reshape(self.shape) ).reshape(self.shape) ) )        
        return minV
    
    ## Returns whether a point given in global coordinates is inside or outside the subfunnel either for a certain time (if C1_p is given) or at all (default)
    # @param X [_shape-np.array] Point in the state-space in global coordinates
    # @param C1_p [float,None] Time specifying the center of the ellipse representing the funnel for a particular time point. If given it will be checked if X is in this ellipse. If None(default) the whole trajectory will be searched if there is any time point for which this point is in the funnel
    # return [boolean] Inside or outside the funnel
    def globPinF(self, X, C1_p=None):
        return self.globVofP(X, C1_p) <= self.lyapVal
    
    #From the subFunnel, the transitions have to be easily accessible for both, discretised and continuous time
    
    ## Discrete time check. This is the more 'important' one
    def checkT(self, t):
        t=np.array(t); t=t.reshape((t.size,)) 
        assert (np.all(t>=self.tSpanD[0]) and np.all(t<=self.tSpanD[1])), 'Time exceeds funnel boundaries'
        return t
    
    ## Continuous time check -> use the one of the traj
    def checkT2(self, t):
        return self.traj.checkT(t)
    
    ## Create time intervals if needed
    # For the rounded float case
    def treatTransT(self, t):
        t = np.array(t); t=t.reshape((t.size,))
        if t.size == 1 or not self.traj.isCycFlag:
            for k in range(t.size):
                if t[k] in [-np.Inf, -intInf]:
                    t[k] = self.tSpanD[0]
                elif t[k] in [np.Inf, intInf]:
                    t[k] = self.tSpanD[1]
                else:
                    self.checkT(t[k])
            return [t] #Always return as list
        else:
            #Here cyclic wrapping makes sense
            for k in range(t.size):
                if t[k] in [-np.Inf, -intInf]:
                    t[k] = self.tSpanD[0]
                elif t[k] in [np.Inf, intInf]:
                    t[k] = self.tSpanD[1]
                    
            if t[1] <= self.tSpanD[1]:
                self.checkT(t)
                return [t]
            else:
                allT = [np.array([t[0], self.tSpanD[1]])]
                tD = t[1]-t[0]
                if tD > self.tSpanD[1]-self.tSpanD[0]:
                    #This means the whole traj is covered
                    allT = [np.array(self.tSpanD)]
                else:
                    tE = np.remainder(t[1], self.tSpanD[1])
                    allT.append(np.array([self.tSpanD[0], self.tSpanD[0]+tE]))
                for aAllT in allT:
                    self.checkT(aAllT)
                return allT
            
            
            
        
        
    ## Create time intervals if needed
    # For the continuous case
    def treatTransT2(self, t):
        t = np.array(t); t=t.reshape((t.size,))
        t=ground(t)
        allT = self.treatTransT(t)
        allTN = []
        for aAllT in allT:
            allTN.append( guround(aAllT) )
        return allTN
    
    def genericTransCall(self, funcTime, funcTrans, *args):
        if len( args ) != 0:
            allT = funcTime(args[0])
            allTrans = []
            for aAllT in allT:
                allTrans = allTrans + funcTrans(aAllT, *args[1::])
        else:
            allTrans = funcTrans( )#All input arguments are defaulted
        return allTrans
    ## Access all function defined in the transDict
    def __getattr__(self, name, *args):
        if name in ['startIn', 'stopIn', 'boundedBy', 'availableAt', 'availableAtTo']:
            #access rounded float functions of transdict
            return lambda *args : self.genericTransCall( self.treatTransT, getattr(self.transDict, name), *args )
        elif name in ['startIn2', 'stopIn2', 'boundedBy2', 'availableAt2', 'availableAtTo2']:
            #access rouded float functions of transdict
            return lambda *args : self.genericTransCall( self.treatTransT2, getattr(self.transDict, name), *args )
        else:
            #pass
            raise KeyError('Function {0} could not be found!'.format(name))
    
    def __deepcopy__(self, *args):
        copyFun = subFun()
        copyFun.__dict__.update(dp(self.__dict__))
        return copyFun
    
    ## Create evaluable invariant strings from myEvalExpr list
    def createInvariants(self):
        self.invariantList = None
        self.invariantListPLOT = None
        for aInv in self.invariantListExpr:
            pltExpr = aInv.getStr(atype = 'plot')
            evalExpr = aInv.getStr(atype = 'eval')
            self.invariantListPLOT = self.invariantListPLOT+' and '+evalExpr if self.invariantListPLOT!=None else pltExpr
            self.invariantList = self.invariantList+' and '+evalExpr if self.invariantList!=None else evalExpr
        return 0
            
  
#####################
## Creates a system of subfunnels (@ref subFun) from a given nominal trajectory (e.g. @ref analyticTraj), a dynamical system (e.g. @ref dynSys) system and targetAlphas. 
# @page funnel1
class funnel1(object):
    ## Constructor
    #
    # once the subfunnelsystem is entirely parameterized by the constructor arguments, it is directly generated and can then be added to
    # a @ref timedAutomata for example.
    # @param basetrajIn [@ref analyticTraj] Nominal trajectory of the subfunnelsystem. Its alphaTime is 1.0.
    # @param dynSysIn [@ref dynSys] Dynamical system whose control can be parameterized by QR-matrices (@ref QRstruct)
    # @param Qstructorlist [list of QRstruct lists; QRstruct] QRstructs to parameterize the control of the dynamical system. If a @ref QRstruct is given then the same parameters will be used for each subfunnel. If a list of list is given it has to match the dimensions of the subfunnel system.
    # @param targetAlphas [list of floats] Specifies the desired alpha for the different speed levels. Since its a time dilatation coefficient, alphaTime=1.1 is equal to a 10% slower speed compared to alphaTime=1.0
    # @param numVelFun [integer] Number of speed levels with the subfunnel system. If the reverseFlag is True, then this number will be doubled.
    # @param numSizeFun [integer] Number of subfunnel with the same velocity/alphaTime but different funnel sizes (in terms of their lyapunov value)  
    # @param reverseFlag [boolean] Flag whether alphaTime can become naegative, corresponding to a change of direction
    # @param conSubFunFlag [boolean] Flag whether the different velocity levels shall be reachable or not. 
    # @param contrcCoeff [float] Contraction coefficient between two consecutive subfunnel of the same velocity. The lyapval of the smaller funnel is contracCoeff*lyapVal of the bigger funnel.
    # \todo conSubFunFlag is not taken into account yet
    # \todo Negative alphaTime is not yet supported
    def __init__(self, baseTrajIn, dynSysIn, Qstructorlist, targetAlphasIn=None, numVelFun=4, numSizeFun=2, reverseFlag = False, conSubFunFlag=True, contracCoeff=0.1, dLfIn = 0.001, tScale=1):
        ###Create a funnel instance
        #baseTrajIn: Reference nominal trajectory in the state space of the dynSysIn
        #dynSysIn: Dynamical system that is to be controlled 
        #Check if the dynamical system is able to follow the given trajectory
        assert np.min(dynSysIn.X_dotLim > baseTrajIn.getMaxXdot()), 'baseTrajectory is not feasible for the given dynamical system'
        self.baseTraj = baseTrajIn
        self.dynSys = dynSysIn
        self.numSubFun = (numVelFun, numSizeFun)
        if isinstance(Qstructorlist, QRstruct):
            self.QRlist = numVelFun*[numSizeFun*[Qstructorlist]]
        else:
            assert ( (len(Qstructorlist) == numVelFun) and (len(Qstructorlist[0]) == numSizeFun)), 'QR has to be given either as a struct (than the same QR is used for all subfunnels) or as a list of list with QR structs as content '
            self.QRlist = Qstructorlist
        
        self.reverseFlag = reverseFlag
        self.conSubFunFlag = conSubFunFlag
        self.contracCoeff = contracCoeff #Factor between the max 'energy' of the Lyapunov functions; F_i = V(X) <= alpha -> F_i+1 = V(X)<=contracCoeff*alpha
        #Handle targetAlphas
        if targetAlphasIn is None:
            #Linear mapping
            targetAlphasIn = [1.0]
            for k in range(1,numVelFun):
                targetAlphasIn.append( targetAlphasIn[-1]*1.25 ) #Each level is 25 percent slower than the last
        self.targetAlphas = gToNearest( np.array(targetAlphasIn)*self.baseTraj.alphaTime )
        
        #Create a list where the dynamic possibly changes
        self.tSpan = self.baseTraj.tSpan
        
        #Create a list of lists holding all subfunnels
        if not self.reverseFlag :
            allVelFun = numVelFun
            #self.funnelSys = numVelFun*[numSubFun*[[]]]
        else:
            allVelFun = 2*numVelFun
            #self.funnelSys = (2*numVelFun)*[numSubFun*[[]]]
        self.funnelSys = [[[] for i in range(numSizeFun)] for j in range(allVelFun)] #Ensure that each sublist has its own id TBD check if implementing deepcopy is necessary for each class
        #Also set links into a dictionnary
        self.funnelDict={}
        
        #Set a search parameters
        self.dLf = dLfIn

        self.tScale = tScale

        #Create the funnel system
        self.createSubFunnelSys()
        
        #Calculate the possible transitions and store them as a dictionnary in each funnel
        self.calcAutoTransitions(self.tScale)
        
        #Check if the constructed the constructed funnels satisfy the all implied constraints
        #This basically concerns backward compatibility
        #self.checkFunnelSys() #TBD: checkUp
        
    ######
    def createSubFunnelSys(self):
        ###Create all velocity and level sub funnels
        for i in range(self.numSubFun[0]):
            for j in range(self.numSubFun[1]):
                self.funnelSys[i][j] = self.createSubFunnel(i,j)#As list of lists
                self.funnelDict[self.funnelSys[i][j].ID] = self.funnelSys[i][j]#As dict
        
        if self.reverseFlag:
            self.createBackFunnels()
        return 0
    ######
    def createBackFunnels(self):
        #Create symmetric funnels with negative alphaTime
        for i in range(self.numSubFun[0]):
            k = 2*self.numSubFun[0]-1-i
            for j in range(self.numSubFun[1]):
                self.funnelSys[k][j] = dp(self.funnelSys[i][j]) #TBD attention check if copy sufficient to reduce unnecessary copying
                tF = self.funnelSys[k][j]
                tF.alphaTime = -1.0*tF.alphaTime
                tF.tSpan = self.tSpan
                tF.dynSys.setALphaTime(tF.alphaTime)
                #Do sth smart with ID and parents:
                tF.ID = -tF.ID
                if not tF.pFunnel.ID == 'base':
                    tF.pFunnel = self.funnelDict(-tF.pFunnel.ID)
                #Add to dict
                self.funnelDict[tF.ID] = tF
        return 0
    ######
    def createSubFunnel(self, i, j):
        ###Create the i,j subfunnel; i==velocity j==level
        imax = self.numSubFun[0]-1; jmax = self.numSubFun[1]-1
        if i==0 and j==0:
            #First funnel to be created; no constraints beside the ones on Xdot
            tF = self.createConstraintFunnel(self.QRlist[i][j].Q, self.QRlist[i][j].R, None, None, self.targetAlphas[0], None)
        elif i>0 and j==0:
            if self.conSubFunFlag:
                #First funnel of this velocity -> Connect it to the smallest funnel of the last velocity level
                tF = self.createConstraintFunnel(self.QRlist[i][j].Q, self.QRlist[i][j].R, self.funnelSys[i-1][jmax], self.funnelSys[i-1][jmax], self.targetAlphas[i], None, self.funnelSys[i-1][0])
            else:
                #If reachability does not have to be assured just generate the biggest funnel possible
                tF = self.createConstraintFunnel(self.QRlist[i][j].Q, self.QRlist[i][j].R, None, None, self.targetAlphas[i], None)
        else:
            #In all the other cases contract the parent funnel
            tF = self.createConstraintFunnel(self.QRlist[i][j].Q, self.QRlist[i][j].R, self.funnelSys[i][j-1], None, False, self.contracCoeff )
        return tF
    ######       
    def createConstraintFunnel(self, Q, R, pFunnel, connectToFunnel, targetAlpha, contracCoeff, connectToFunnelBack=None):
        #traj is the starting guess to create the funnel
        #connectToFunnel is like a constraint
        #changeSpeed is a boolean. If True the time dilatation for traj can be adjusted
        #LaypTuple contains the maximum Lyapfunction value and the correspond P to calculate it
        #its either connectToFunnel+changeSpeed(Different Velocity desired) or LyapTuple(Create sublevel Funnel) or none of both -> maximise and respect sys constraints
        #connectToFunnelBack can be used to ensure (at least almost) backwards compatibility; #TBD:CheckUp still needs to be implemented
        tF = subFun()
        tF.dLf = self.dLf
        
        if connectToFunnel==None and isinstance(targetAlpha, float) and contracCoeff==None:
            #Copy trajectory and directly use the first targetAlpha
            tF.traj = dp(self.baseTraj)
            tF.alphaTime = targetAlpha#self.baseTraj.alphaTime
            
            tF.dynSys = dp(self.dynSys)
            #Not necessary here since system completely linear but kept for generality
            thisT = 0.5*(tF.tSpan[1]+tF.tSpan[0])
            thisX = tF.traj.getX(thisT)
            trajXdotMax = tF.traj.getMaxXdot([tF.tSpan[0],tF.tSpan[1]])
            tF.dynSys.setQ( Q )
            tF.dynSys.setR( R )#Control is now calculated
            dynSysXdotMax = tF.dynSys.getXdotLim(thisX)#Approximation by midpoint. TBD:perform a min search? -> No necessary for lin sys
            tF.dXdotmax = (dynSysXdotMax-trajXdotMax).reshape(tF.shape)
            if namin(tF.dXdotmax.squeeze()) <= 0:
                raise ValueError('Reference trajectory derivative exceeds system capacities')
            #Calculate the corresponding minimal Lyapunov value and restrain the funnnel to this value
            #[~,wa] = (tF.dynSys.getAbar()), sort='descending', type='normed')
            #[~,wp] = eig(tF.dynSys.getP()), 'ascending')
            #tF.lyapVal = (nmin(tF.dXmax)/wa[0])**2.0*wp[0]
            ###less (but still) conservative alternative
            #[E,V] = eig(tF.dynSys.getP())
            #Xdot = Abar*X; X_allowed = E*V*(-0.5)*sqrt(alpha)*u with u:unitary vector; Xdot_max = Abar*E*V*(-0.5)*sqrt(alpha)*u;
            #Xdot_max(i) can not get bigger than max(abs(Abar2[i,:]))*sqrt(alpha) with Abar2 = Abar*E*V*(-0.5); Get alpha for each row and take smallest
            #tF.lyapVal = (float(namin(div(tF.dXdotmax.squeeze(), namax( nabs(dot(tF.dynSys.getAbar(), getT(tF.dynSys.getP()))), axis = 1).squeeze()),0)))**2.0
            tF.lyapVal = (float(namin(div(tF.dXdotmax.squeeze(), norm( dot(tF.dynSys.getAbar(), getT(tF.dynSys.getP())), axis = 1).squeeze()),0)))**2.0 #The old one was actually not conservative, this one is
            
            tF.ID = 0;
            tF.parentFunnel = None
            #This Funnel is now completely specified
        
        elif connectToFunnel==None and targetAlpha==False and isinstance(contracCoeff, float):
            #This section creates a subfunnel with the same speed and trajectory as the last one but with a different
            #'energy' (in terms of lyapunov function) level
            #TBD this function might be changed to directly consider compatibility with speed changes
            tF.parentFunnel = pFunnel #Set link
            tF.ID = pFunnel.ID+1
            
            tF.traj = dp(pFunnel.traj) #No need to copy TBD sure?
            
            tF.dynSys = dp(self.dynSys)
            thisT = 0.5*(tF.tSpan[1]+tF.tSpan[0])
            thisX = tF.traj.getX(thisT)
            
            trajXdotMax = tF.traj.getMaxXdot([tF.tSpan[0],tF.tSpan[1]])
            #One might want to modify the QR setting for the smaller funnel in order to increase convergence when close TBD
            tF.dynSys.setQ( Q )
            tF.dynSys.setR( R )#Control is now calculated
            dynSysXdotMax = tF.dynSys.getXdotLim(thisX)#Approximation by midpoint. TBD:perform a min search? -> No necessary for lin sys
            tF.dXdotmax = (dynSysXdotMax-trajXdotMax).reshape(tF.shape)
            if namin(tF.dXdotmax.squeeze()) <= 0:
                raise ValueError('Reference trajectory derivative exceeds system capacities')
            #Calculate a Lyapunov value such that the possible states satisfy both, the contraction and the limits
            #on dXdot. After that the necessary convergence time is calculated
            T2u = getT(tF.dynSys.getP())
            P3 = ndot( T2u.T, pFunnel.dynSys.getP(), T2u ) / (pFunnel.lyapVal*contracCoeff)
            #[_unused,V3] = eig(P3, 'descending', discardImag=True) #shit
            [_unused,V3] = eig(P3, discardImag=True) #shit
            lyapVal1 = min(1.0/V3) #Now the new funnel is inside the contracted  parent funnel
            #Check if this Funnel is acceptable considering dXdotmax limits
            #lyapVal2 = (float(namin(div(tF.dXdotmax.squeeze(), namax( nabs(dot(tF.dynSys.getAbar(), getT(tF.dynSys.getP()))), axis = 1).squeeze()),0)))**2.0
            lyapVal2 = (float(namin(div(tF.dXdotmax.squeeze(), norm( dot(tF.dynSys.getAbar(), getT(tF.dynSys.getP())), axis = 1).squeeze()),0)))**2.0 #The old one was actually not conservative, this one is
            #Take the minimum to ensure safety (If QR unchanged, lyapVal1 should be smaller than lyapVal2)
            if lyapVal2 < lyapVal1: print('New QR together setting with Xdot limits of the dynamical system limits funnel size more than the contracted parent funnel!')
            tF.lyapVal = min([lyapVal1, lyapVal2])
            
            T1u = getT(pFunnel.dynSys.getP())
            P4 = dot(dot(T1u.T, tF.dynSys.getP()), T1u)/tF.lyapVal
            #[_unused,V4] = eig(P4, 'descending')#shit
            [_unused,V4] = eig(P4, discardImag=True)
            targValParent = min(1.0/V4) #Now all states are inside the new funnel in turn being inside the contracted  parent funnel
            
            #Calculate a conservative limit for the convergence of the funnel
            #1 Calculate the derivative of the lyapunov function            
            Q_hat = pFunnel.dynSys.getLyapDot()#TBD: adjust for nonlinear systems
            #Check for the point with the wost convergence for the given lyapfunction
            Tp = getT(pFunnel.dynSys.getP())#Normalized transformation
            Q_hat = ndot(Tp.T, Q_hat, Tp)
            [Ec,Vc] = eig(Q_hat, discardImag=True)
            assert max(Vc) < 0, 'Lyapunov function has non-negative derivative'
            #Get smallest eigenvalue
            #Ec=Ec[:,0].reshape((pFunnel.dynSys.stateDim,1))
            #Vc=Vc[0]
            #pFunnel.lyapValDotminu=float(ndot(Ec.T, Q_hat, Ec).real)#The actual minimal derivative of the lyapunov value depends linearly on the the scaling. -> Vminreal = lyapValDotminu*lyapVal 
            pFunnel.lyapValDotminu=max(Vc)
            pFunnel.PtoCmaxConvTime = float(1.0/pFunnel.lyapValDotminu*ln(targValParent/pFunnel.lyapVal)) #Exponential decrease of lyapVal with conservative minimal speed
            
            ###Finished creating sublevel funnel
        
        elif isinstance(connectToFunnel, subFun) and isinstance(targetAlpha, float) and contracCoeff==None:
            #This function creates a funnel with a different speedlevel and transitions are ensured from connecttofunnel and thisfunnel 
            #They are always possible
            tF.parentFunnel = None#Base funnel of this velocity  
            tF.ID = pFunnel.ID+1
            tF.traj = dp(pFunnel.traj)
            
            tF.dynSys = dp(self.dynSys)
            #TBD: Change this for nonlinear systems
            thisT = 0.5*(tF.tSpan[1]+tF.tSpan[0])
            thisX = tF.traj.getX(thisT)
            
            #Perform calculations in order to arrive the closest possible to the target speed while being connected to
            # the connectToFunnel and respecting the constraints.
            #1:Calculate the limits of the dynamical system
            #One might want to modify the QR setting for the smaller funnel in order to increase convergence when close TBD
            tF.dynSys.setQ( Q )
            tF.dynSys.setR( R )#Control is now calculated
            dynSysXdotMax = tF.dynSys.getXdotLim(thisX)#Approximation by midpoint. TBD:perform a min search? -> No necessary for lin sys
            
            #2:Reduce alphaTime and check each time if the limits on Xdot are respected
            Tcon = getT(connectToFunnel.dynSys.getP(), connectToFunnel.lyapVal) #Uns
            thisT = getT(tF.dynSys.getP())
            thisTi = inv(thisT) 
            #TBD search better method then brute force
            check = True
            lVF=lV=0; #Dummies
            aTF=aT=dp(tF.alphaTime)
            #Search parameter
            epsT = 0.001*(aT-targetAlpha)
            while not ground(gToNearest(gToNearest2(epsT))) == ground(gToNearest(epsT)):
                epsT = gToNearest(epsT+guround(1))#0.001*(aT-targetAlpha)
            #'Additional' lyapunov value needed to cover the ellipse of the small parent funnel
            [_unused, sv, _unused] = svd(dot(thisTi, Tcon))
            lVadd = namax(sv)#TBD:Recheck with different QR
            
            #Needed terms for assumed backward compatibility
            if isinstance(connectToFunnelBack, subFun):
                #The smallest subfunnel of this velocity needs needs to be connected to the biggest subFunnel of the last velocity
                transFromu = getT( tF.dynSys.getP() )
                transToui = inv(getT(connectToFunnelBack.dynSys.getP()))
                [_unused, svu, _unused] = svd(dot(transToui, transFromu))
                lVaddBacku = namax(svu)#TBD:check with different QR; conservative minimum additional value to cover to-funnel ellipse #Unscaled version since the necessary lyapVal is being determined
                rootlVT = connectToFunnelBack.lyapVal**0.5
            
            #Calculate a sample of the positions in state space
            #normalize trajectory
            toTraj = connectToFunnel.traj            
            sT = toTraj.unitToTime( np.linspace(0.0, 1.0, toTraj.N_searchTo) )#TBD check if scanning the search points is sufficient to ensure many transitions
            sP = toTraj.getX(sT)
            oldtDF = toTraj.tDFc
            while ( ground( np.fabs(aT - targetAlpha))>=ground(np.fabs(epsT)) ) and check:
                #Create a dummy traj to avoid rounding errors
                dtF = dp(tF)
                #Update final values
                lVF = lV; aTF = aT;
                #Reduce alphaTime
                aT -= epsT
                #ensure that changing the alpha results in a trajectory that can still be correctly discretized
                aT = gToNearest2(aT)
                #Set new alpha and get maximum allowed lyap val due to constraints lVc
                dtF.alphaTime = aT #Setter is now decorate so this should be valid
                trajXdotMax = dtF.traj.getMaxXdot() #Elementwise maximum
                dXdotmax = (dynSysXdotMax-trajXdotMax).reshape(dtF.shape)
                #lVc = (float(namin(div(dXdotmax.squeeze(), namax( nabs(dot(tF.dynSys.getAbar(), getT(tF.dynSys.getP()))), axis = 1).squeeze()),0)))**2.0
                #lVc = (float(namin(div(dXdotmax.squeeze(), namax( nabs(dot(dtF.dynSys.getAbar(), thisT)), axis = 1).squeeze()),0)))**2.0
                lVc = (float(namin(div(dXdotmax.squeeze(), norm( dot(dtF.dynSys.getAbar(), thisT), axis = 1).squeeze()),0)))**2.0 #The old one was actually not conservative, this one is
                #lVc = (float(namin(div(dXdotmax, namax( nabs(dot(tF.dynSys.getAbar(), getT(tF.dynSys.getP()))), axis = 1)),0)))**2
                #Now calculate the minimum lyapVal needed to connect them
                #One has to note that the 'position' is the same and that only the derivatives change (poisition equal since the clocks will be chosen this way)
                #If the velocities change 'a lot' it might be true that other position are closer (in the sense of the P norm) but this is not considered since we want
                #updates of the from: transGuard=True; transSet C1_c=alpha*C1_p
                DtDF = oldtDF - dtF.traj.tDFc
                lV = -np.Inf
                for k in range(sP.shape[1]):
                    thisV = dot(thisTi, mult(DtDF, sP[:,k]).reshape(dtF.shape)).reshape(dtF.shape)
                    lV = max(lV, norm(thisV))
                #Add the additional lyap value and square
                lV = (lV+lVadd)**2.0
                
                #If the calculate lyapunov value is no longer feasible uncheck
                if lV > lVc:
                    check = False
                
                #Do the additional backward compatibility check
                if isinstance(connectToFunnelBack, subFun):
                    for k in range(sP.shape[1]):
                        thisVal = norm( dot(transToui, mult(DtDF, sP[:,k]).reshape(dtF.shape)).reshape(dtF.shape) )
                        check = check and ( rootlVT > 1.1*(lVaddBacku*((lV*self.contracCoeff**(self.numSubFun[1]-1)))**0.5+thisVal) )#Scale the necessary additional value with the contracted current lyapVal
                        
                #Done #TBD: Rewrite this part of code its pretty ugly
            #Check
            if np.fabs((aTF-targetAlpha)) > np.fabs(2*epsT): print('Failed to attain desired  alphaTime = {0}. Resulting alphaTime = {1}.'.format(targetAlpha, aTF)) 
            #Save the calculated values
            tF.alphaTime = aTF
            #Rather impose the maximal funnel size
            #tF.lyapVal = lVF
            #TBD: find a smarter search algo (test first if the speed is achievable for example)
            lVc = (float(namin(div(dXdotmax.squeeze(), namax( nabs(dot(tF.dynSys.getAbar(), thisT)), axis = 1).squeeze()),0)))**2.0
            tF.lyapVal = lVF + min((1.0-(lVc-lVF)**0.5/lVc**0.5),0.5)*(lVc-lVF)
            #if lVc > lVF+0.25*(lVc-lVF):
                #tF.lyapVal = lVF+0.25*(lVc-lVF)
            #else:
                #tF.lyapVal = lVc
            
            
        else:
            raise ValueError('The inputs did not correspond to one of the specified cases')
        return tF

        ############
    def calcAutoTransitions(self, tScale=1):
        #Create all the transitions possible for each subfunnel to other subfunnels
        #Start of with the connection between the different levels at the same velocity
        #subfunnel; i==velocity j==level
        #Each funnel has two clocks C1_p/c = gives the location in the parent/child funnel and C2_p indicating the duration (real time) in the funnel 
        #Storing the conditions and setting as a string
        #Transitions are stored in a dict using the a in [a,b]x{c} as key
        #if the key is taken, than a list will be stored holding all the transitions there
        #Transitions that can 'always' be accessed are stored at zero
        if tScale!=np.floor(tScale) or tScale!=np.ceil(tScale):
            warnings.warn("Using a custom scaling which is not an integer makes the approach non-conservative", UserWarning)

        up = lambda x: int(np.ceil(x * tScale))
        down = lambda x: int(np.floor(x * tScale))

        for i in range(self.numSubFun[0]):
            for j in range(self.numSubFun[1]):
                tF = self.funnelSys[i][j]
                #Set up a list of invariants
                tF.invariantListExpr.append( myEvalString( ['<=', down(tF.tSpanD[0]), 'C1_p'] ) ) #Invariants ensuring that the length of the funnel is not exceeded
                tF.invariantListExpr.append( myEvalString( ['<=', 'C1_p', up(tF.tSpanD[1])] ) )
                #tF.invariantList.append( '{0:.12e}<=C1_p and C1_p<={1:.12e}'.format(tF.tSpan[0], tF.tSpan[1]) ) 
                #tF.transDict = transitionDict()
                if j < self.numSubFun[1]-1:
                    #Funnel has a contracted one inside
                    #tF.transDict[tF.tSpan] = transStruct(tF.ID, self.funnelSys[i][j+1].ID, 'C2_p>={0:.12e}'.format(tF.PtoCmaxConvTime), 'C1_c=C1_p; C2_c=0.0',tF.tSpan, transGuardPLOTIn='C2_p>={0:.4f}'.format(tF.PtoCmaxConvTime))
                    #tF.transDict[tF.tSpanD] = transStruct(tF.ID, self.funnelSys[i][j+1].ID, tF.tSpanD, [['>=', 'C2_p', up(np.ceil(tF.PtoCmaxConvTime*globalRoundVal))]], [['=', 'C1_c', 'C1_p'], ['=', 'C2_c', 0.0]] )
                    # Tchecker fix
                    tF.transDict[tF.tSpanD] = transStruct(tF.ID, self.funnelSys[i][j + 1].ID, tF.tSpanD, [['>=', 'C2_p', up(np.ceil(tF.PtoCmaxConvTime * globalRoundVal))]], [['=', 'C2_c', 0.0]])
                    
                if tF.traj.isCycFlag:
                    #Add a transition from the end to the beginning to be able to deal with the invariant
                    #tF.transDict[(tF.tSpan[1],tF.tSpan[1])] = transStruct(tF.ID, tF.ID, 'C1_p=={0:.12e}'.format(tF.tSpan[1]), 'C1_c={0:.12e}; C2_c=C2_p'.format(tF.tSpan[0]), (tF.tSpan[1],tF.tSpan[1]), transGuardPLOTIn='C1_p=={0:.4f}'.format(tF.tSpan[1]), transSetPLOTIn='C1_c={0:.4f}; C2_c=C2_p'.format(tF.tSpan[0]) )
                    #tF.transDict[(tF.tSpanD[1],tF.tSpanD[1])] = transStruct(tF.ID, tF.ID, (tF.tSpanD[1], tF.tSpanD[1]), [['==', 'C1_p', up(tF.tSpanD[1])]], [['=', 'C1_c', down(tF.tSpanD[0])], ['=', 'C2_c', 'C2_p']] )
                    # Tchecker fix to avoid local=local
                    tF.transDict[(tF.tSpanD[1], tF.tSpanD[1])] = transStruct(tF.ID, tF.ID, (tF.tSpanD[1], tF.tSpanD[1]),
                                                                             [['==', 'C1_p', up(tF.tSpanD[1])]],
                                                                             [['=', 'C1_c', down(tF.tSpanD[0])]])
                    #Add two more rough transitions to represent the cyclic transition for simulation (Transition is not caught due to numerical issues)
                    #TBD find a better way to do this
                    #tF.transDict[(tF.tSpan[1]-0.0001*(tF.tSpan[1]-tF.tSpan[0]),tF.tSpan[1])] = transStruct(tF.ID, tF.ID, '{0:.12e}<=C1_p and C1_p<={1:.12e}'.format(tF.tSpan[1]-0.0001*(tF.tSpan[1]-tF.tSpan[0]),tF.tSpan[1]), 'C1_c={0:.12e}; C2_c=C2_p'.format(tF.tSpan[0]), (tF.tSpan[1]-0.0001*(tF.tSpan[1]-tF.tSpan[0]),tF.tSpan[1]), transGuardPLOTIn='{0:.4f}<=C1_p and C1_p<={1:.4f}'.format(tF.tSpan[1]-0.0001*(tF.tSpan[1]-tF.tSpan[0]),tF.tSpan[1]), transSetPLOTIn='C1_c={0:.4f}; C2_c=C2_p'.format(tF.tSpan[0]) )
                    #tF.transDict[(tF.tSpan[0],tF.tSpan[0]+0.0001*(tF.tSpan[1]-tF.tSpan[0]))] = transStruct(tF.ID, tF.ID, '{0:.12e}<=C1_p and C1_p<={1:.12e}'.format(tF.tSpan[0],tF.tSpan[1]+0.0001*(tF.tSpan[1]-tF.tSpan[0])), 'C1_c={0:.12e}; C2_c=C2_p'.format(tF.tSpan[0]), (tF.tSpan[0],tF.tSpan[0]+0.0001*(tF.tSpan[1]-tF.tSpan[0])), transGuardPLOTIn='{0:.4f}<=C1_p and C1_p<={1:.4f}'.format(tF.tSpan[0],tF.tSpan[0]+0.0001*(tF.tSpan[1]-tF.tSpan[0])), transSetPLOTIn='C1_c={0:.4f}; C2_c=C2_p'.format(tF.tSpan[0]) )
                    
                    
        if self.reverseFlag:
            raise NotImplementedError('This has to be modified')
            for i in range(2*self.numSubFun[0]-1, self.numSubFun[0], -1):
                for j in range(self.numSubFun[1]):
                    #Add the symmetric reverse funnels
                    tF = self.funnelSys[i][j]
                    #Also set up a list of invariants
                    tF.invariantList.append( '{0:.12e}<=C1_p and C1_p<={1:.12e}'.format(tF.tSpan[0], tF.tSpan[1]) ) #Invariant ensuring that the length of the funnel is not exceeded
                    #tF.transDict = transitionDict()
                    if j > self.numSubFun[1]-1:
                        tF.transDict[(tF.tSpan)] = transStruct(tF.ID, self.funnelSys[i][j-1].ID, 'C2_p>={0:.12e}'.format(tF.PtoCmaxConvTime), 'C1_c=C1_p; C2_c=0.0',tF.tSpan, transGuardPLOTIn='C2_p>={0:.4f}'.format(tF.PtoCmaxConvTime))
                    if tF.traj.isCycFlag:
                        #Add a transition from the end to the beginning to be able to deal with the invariant
                        #tF.transDict[(tF.tSpan)] = transStruct(tF.ID, tF.ID, 'C1_p=={0:.12e}'.format(tF.tSpan[1]), 'C1_c={0:.12e}; C2_c=C2_p'.format(tF.tSpan[0]), tF.tSpan, transGuardPLOTIn='C1_p=={0:.4f}'.format(tF.tSpan[1]), transSetPLOTIn='C1_c={0:.4f}; C2_c=C2_p'.format(tF.tSpan[0]) )
                        #Tchecker fix
                        raise NotImplementedError
                    
        #Transitions to enter a different velocity level
        #TBD check if one wants to add other options
        #Fast to Slow
        #Fast skip
        #return 0
        if self.numSubFun[1] >= 2:
            #subfunnel: i==velocity j==level
            #===================================================================
            #Changing velocity is no longer a special case; transitions set of the type t'=a*t+b with a!=1 and b!=0 risk to be indecidable
            #fKeys = self.funnelDict.keys()
            #for fromKey in fKeys:
            #    fromFun = self.funnelDict[fromKey]
            #    for toKey in fKeys:
            #        if fromKey==toKey:break#No auto-transitions
            #        toFun = self.funnelDict[toKey]
            #        thisTrans = getPossibleTrans_vC(fromFun, toFun)#Calculate transitions
            #        for aTrans in thisTrans:
            #            fromFun.transDict[aTrans.tSpan] = aTrans
            for iFrom in range(self.numSubFun[0]):
                for iTo in range(self.numSubFun[0]):
                    for jFrom in range(self.numSubFun[1]):
                        for jTo in range(self.numSubFun[1]):
                            #Check for transitions between different speed-levels
                            #if iFrom == iTo: continue
                            #Check for transitions between different subfunnels
                            if (iFrom == iTo and jFrom == jTo): continue
                            fromFun = self.funnelSys[iFrom][jFrom]
                            toFun = self.funnelSys[iTo][jTo]
                            thisTrans = getPossibleTrans(fromFun, toFun, fromFun.dLf)#Calculate transitions
                            print(str(len(thisTrans))+' transitions exist between '+str(fromFun.ID)+' ; '+str(toFun.ID))
                            for aTrans in thisTrans:
                                fromFun.transDict[aTrans.tSpan] = aTrans
                            
                    
                
            #Done
                    
                    
            
            # for i in range(self.numSubFun[0]-1):
            #     #Connect biggest and smallest (transition ensured by construction)
            #     jm = self.numSubFun[1];
            #     #small to big; fast to slow
            #     #TBD : check up if there is sense in allowing a transition interval
            #     self.funnelSys[i][jm].transDict[self.funnelSys[i][jm].tSpan] = transStruct( self.funnelSys[i][jm].ID, self.funnelSys[i+1][0].ID, 'True', 'C1_c={0:.12e}*C1_p; C2_c=0.0'.format(self.funnelSys[i+1][0].traj.alphaTime / self.funnelSys[i][jm].traj.alphaTime),self.funnelSys[i][jm].traj.tSpan, transSetPLOTIn = 'C1_c={0:.4f}*C1_p; C2_c=0.0'.format(self.funnelSys[i+1][0].traj.alphaTime / self.funnelSys[i][jm].traj.alphaTime)) 
            #     #small to big; slow to fast
            #     self.funnelSys[i+1][jm].transDict[self.funnelSys[i+1][jm].tSpan] = transStruct( self.funnelSys[i+1][jm].ID, self.funnelSys[i][0].ID, 'True', 'C1_c={0:.12e}*C1_p; C2_c=0.0'.format(self.funnelSys[i][0].traj.alphaTime / self.funnelSys[i+1][jm].traj.alphaTime),self.funnelSys[i+1][jm].traj.tSpan, transSetPLOTIn = 'C1_c={0:.4f}*C1_p; C2_c=0.0'.format(self.funnelSys[i+1][0].traj.alphaTime / self.funnelSys[i][jm].traj.alphaTime)) 
            # 
            # if self.reverseFlag:
            #     raise NotImplementedError('This has to be modified')
            #     for i in range(2*self.numSubFun[0]-1, self.numSubFun[0]+1, -1):
            #         jm = self.numSubFun[1];
            #         #small to big; fast to slow
            #         #TBD : check up if there is sense in allowing a transition interval
            #         self.funnelSys[i][jm].transDict[self.funnelSys[i][jm].tSpan] = transStruct( self.funnelSys[i][jm].ID, self.funnelSys[i-1][0].ID, 'True', 'C1_c=C1_p; C2_c=0.0',tF.tSpan) 
            #         #small to big; slow to fast
            #         self.funnelSys[i-1][jm].transDict[self.funnelSys[i-1][jm].tSpan] = transStruct( self.funnelSys[i-1][jm].ID, self.funnelSys[i][0].ID, 'True', 'C1_c=C1_p; C2_c=0.0',tF.tSpan)
            #     #Allow switching the direction
            #     #small to big; forward to backward
            #     #TBD : check up if there is sense in allowing a transition interval
            #     self.funnelSys[self.numSubFun[0]-1][jm].transDict[self.funnelSys[self.numSubFun[0]-1][jm].tSpan] = transStruct( self.funnelSys[self.numSubFun[0]-1][jm].ID, self.funnelSys[self.numSubFun[0]][0].ID, 'True', 'C1_c=C1_p; C2_c=0.0',tF.tSpan) 
            #     #small to big; backward to forward
            #     self.funnelSys[self.numSubFun[0]][jm].transDict[self.funnelSys[self.numSubFun[0]][jm].tSpan] = transStruct( self.funnelSys[self.numSubFun[0]][jm].ID, self.funnelSys[self.numSubFun[0]-1][0].ID, 'True', 'C1_c=C1_p; C2_c=0.0',tF.tSpan)
            #===================================================================
        return 0    
        ######
        
    def checkFunnelSys(self):
        pass
        return 0
        
    ###Done
#####################

## A timedAutomata consists of multiple subfunnel systems (@ref funnel1) and generates all the transitions between the different subfunnels.
# 
# @page timedAutomata
class timedAutomata(object):
    ## timedAutomata creates an instance of a funnel based timed automata.
    #
    # It creates the transition guards and states used by uppaal for instance
    # and use can be used for simulation
    def __init__(self):
        self.funnelSysList = [] 
        self.funnelDict = {}
        self.XIn = None
    
    ## Add a funnelsystem to the list of funnel systems
    #
    # @param aFunnel [@ref funnel1] Funnel system to be added
    def addFunnel(self, aFunnel):
        self.funnelSysList.append(aFunnel)
    
    ## A helper function that makes the IDs of subfunnels unique within the timed automat
    def uniqueID(self):
        #This must be called before calcTransitions!
        #It converts the numerical funnel ID into a string and adds a lowercase letter prefix to make it unique
        #Also store a link to all the funnels in a dict
        
        alphab = string.ascii_lowercase
        N=len(alphab)
        n=1
        while N**n <= len(self.funnelSysList):
            n+=1
        
        #Replace the ID
        numS=0
        for aFunSys in self.funnelSysList:
            indList = n*[[]]
            prefix = ''
            num=numS
            for l in range(n-1,-1,-1):
                indList[l] = num%(N**(l+1))
                num = int(np.floor(num/N**(l+1)))
                prefix = prefix+alphab[indList[l]]
            numS+=1
            aFunSys.ID = prefix
            for _unused, aFun in aFunSys.funnelDict.items():
                #Replace ID
                aFun.ID = prefix+'{0:d}'.format(aFun.ID)
                #And all the IDs in the transitions
                for _unused, aTransL in aFun.transDict.startDict.items():
                    for aTrans in aTransL:
                        aTrans.parentID = prefix+'{0:d}'.format(aTrans.parentID)
                        aTrans.childID = prefix+'{0:d}'.format(aTrans.childID)
                #Add to dictionnary
                self.funnelDict[aFun.ID] = aFun
        #Everything renamed
        return 0
    
    def calcTransition(self, tScale=1.):
        if tScale!=np.floor(tScale) or tScale!=np.ceil(tScale):
            warnings.warn("Using a custom scaling which is not an integer makes the approach non-conservative", UserWarning)
        ###Calculate all possible transitions between different funnel sets
        for aFunSysK in self.funnelSysList:
            #Each funnelsys in the funnelsyslist
            for _unused, aFunK in aFunSysK.funnelDict.items():
                #Each subfunnel of this funnelsys
                for aFunSysL in self.funnelSysList:
                    #test against each funnelsys (beside itself)
                    if aFunSysK.ID==aFunSysL.ID: continue
                    for _unused, aFunL in aFunSysL.funnelDict.items():
                        allTrans = getPossibleTrans(aFunK, aFunL, aFunK.dLf, tScale=tScale)#Calculate all possible transitions from aFunK to AFunL
                        #Add them to the transition dict of aFunK
                        print('{0:d} transitions exist between {1} and {2}'.format(len(allTrans), aFunK.ID, aFunL.ID))
                        for aTrans in allTrans:
                            aFunK.transDict[aTrans.tSpan] = aTrans#Using the tSpan as key
    
    ## Create the invariants expressions of all registred subfunnels
    def createInvariants(self):
        for _unused, funnel in self.funnelDict.items():
            funnel.createInvariants()
        return 0
        
    ## Simulate a run given as an initial condition and a timed word
    #
    # This is simulation is performed 'unguarded' meaning that transitions are simply taken and no verification wether a corresponding veryfied transition exists.
    # This simulation is done in continuous time.
    # @page timedAutomata.unguarded_simulate
    # XIn: Initial offset relative to the nominal trajectory given by funIn
    # funIn: ID of the initial funnel
    # clocksIn: Tuple with the current state of the parent funnel clocks = [C1_p, C2_p]
    # timedWord: List of lists: [succ1=[delta T:Time passed since changed into this funnel when transition shall take place, ID:ID of the funnel to go to, [C1_c, C2_c]],....]. If the simulation shall 'simply' continue without changing the funnel, ID can be set to None 
    # dT = time step for output
    def unguarded_simulate(self, XIn, funIn, clocksIn, timedWord, dT, ax = None, cMap='jet', faceAlpha=0.5, dim=[0,1], oneDim=0, lineStyle=['k', '-'], refStyle=['r', '--'], tSampleIn=None, indNotPos = None, tGlobIn = 0.0):    
        #Create some colors
        cNum=0
        if len(cMap)>1 and not isinstance(cMap, list) and not isinstance(cMap, tuple):
            #Try of the given cMap is really a cMap
            cList=pu.getColorList(len(timedWord)+1, cMap)
        else:
            #If not interpret it as a color
            cList=(len(timedWord)+1)*[cMap]
        
        cFun = self.funnelDict[funIn]
        pFun = self.funnelDict[funIn]
        
        C1_p = np.array(clocksIn[0]); C2_p = np.array(clocksIn[1])
        
        m = cFun.dynSys.stateDim
        if funIn =='_stat_':
            newX = np.zeros(m,1)
            cFun.traj = lambda t: newX
        #get current state
        dXc = XIn.reshape((m,1))
        Xc = dXc + self.funnelDict[funIn].traj.getX(C1_p)
        #Plot initial state
        if ax!=None: pu.plotEllipse(ax[0], cFun.traj.getX(C1_p)[dim,:], cFun.dynSys.getP()[np.ix_(dim,dim)] , cFun.lyapVal, color = cList[0], faceAlpha=faceAlpha)
        if ax!=None: ax[0].plot(Xc[dim[0],:], Xc[dim[1],:], '.', color = cList[cNum], markersize=10.0, markerfacecolor = cList[cNum]) 
        
        #Set up return values
        success = True
        allXref = np.zeros((m,0))#Reference position
        allX = np.zeros((m,0))#Absolute position
        allDX = np.zeros((m,0))#Relative position
        allT = np.zeros((0,))#time array
        if ax!=None: ax[0].plot(Xc[dim[0],:], Xc[dim[1],:], '.', color = lineStyle[0], markersize=6.0, markerfacecolor = lineStyle[0]) 
        #Simulate
        tLast = tGlobIn
        for aLetter in timedWord:
                        
            thisT = np.array( list(np.arange(0, aLetter[0], dT)) + [aLetter[0]] )#TBD uff ca fait moche #np.concatenate( (np.arange(0, aLetter[0], dT), np.array(aLetter[0]).squeeze()) )
            thisC1_p = C1_p + thisT
            thisC2_p = thisT
            thisDX = cFun.dynSys.propState(thisC2_p, dXc)
            thisXref = cFun.traj.getX(thisC1_p)
            thisX = thisXref + thisDX
            #plot
            if ax!=None: ax[0].plot(thisXref[dim[0],:], thisXref[dim[1],:], color=cList[cNum])
            
            #Store the results
            allT = np.concatenate((allT, tLast+thisT))
            tLast = allT[-1]
            allXref = np.concatenate((allXref, thisXref), axis = 1)
            allX = np.concatenate((allX, thisX), axis = 1)
            allDX = np.concatenate((allDX, thisDX), axis = 1)
            
            
            #Calculate transition
            Xc = thisX[:,-1].reshape((m,1))
            if ax!=None: ax[0].plot(Xc[dim[0],:], Xc[dim[1],:], '.', color = lineStyle[0], markersize=6.0, markerfacecolor = lineStyle[0]) 
            #Update clocks change funnel and plot
            C1_p+=aLetter[0]
            C2_p+=aLetter[0]
            if aLetter[1] != None:
                newClocks = aLetter[2]
                if newClocks[0] != 'cont':
                    C1_c = newClocks[0]
                else:
                    C1_c = C1_p
                if newClocks[1] != 'cont':
                    C2_c = newClocks[1]
                else:
                    C2_c = C2_p
                
                if aLetter[1] == '_discard_':
                    continue
                cFun = self.funnelDict[aLetter[1]]
                
                if cFun =='_stat_':
                    newX = dp(Xc)
                    newX[np.ix_(indNotPos)]=0.0
                    cFun.traj = lambda t: newX
                
                else:
                    if ax != None: pu.plotTransition2(ax[0], self, pFun, C1_p, cFun, C1_c, color=[cList[cNum], cList[cNum+1]], faceAlpha=faceAlpha)
                
                pFun = cFun
                C1_p=C1_c
                C2_p=C2_c
                
            
            cNum+=1
            #Calculate the current error
            dXc = Xc-cFun.traj.getX(C1_p)    
                
        #Done 'simulation'
        #Plot the result
        if ax!=None:
            ax[0].plot( allX[dim[0],:], allX[dim[1],:], color=lineStyle[0], linestyle=lineStyle[1] )
            ax[1].plot( allT, allXref[oneDim,:], color=refStyle[0], linestyle=refStyle[1] )
            ax[1].plot( allT, allX[oneDim,:], color=lineStyle[0], linestyle=lineStyle[1]  )
            
            #ax[1].plot( allT, np.apply_along_axis(norm, axis=0, arr=allDX) )
        
        if tSampleIn != None:
            #Resample all the points
            allT, allXref, allX, allDX = doSampleInter( tSampleIn, allT, allXref, allX, allDX )
        
        return [success, allT, allXref, allX, allDX, [C1_p, C2_p]]
    
    ## Simulate a run given as an initial condition and a timed word
    #
    # This is simulation is performed 'guarded' meaning that actually varified transitions are searched for each desired transition. The simulation will fail if no corresponding transition can be found.
    # This simulation is done in rounded float time
    # @page timedAutomata.guarded_simulate
    # XIn: Initial offset relative to the nominal trajectory given by funIn
    # funIn: ID of the initial funnel
    # clocksIn: Tuple with the current state of the parent funnel clocks = [C1_p, C2_p]
    # timedWord: List of lists: [succ1=[delta T:Time passed since changed into this funnel when transition shall take place, ID:ID of the funnel to go to, C1_c],....]. If the simulation shall 'simply' continue without changing the funnel, ID can be set to None 
    # dT = time step for output
    def guarded_simulate(self, XIn, funIn, clocksIn, timedWord, dT, ax = None, cMap='jet', faceAlpha=0.5, dim=[0,1], oneDim=0, lineStyle=['k', '-'], refStyle=['r', '--'], tSampleIn=None, tGlobIn = 0.0):    
        #Create some colors
        cNum=0
        if len(cMap)>1 and not isinstance(cMap, list) and not isinstance(cMap, tuple):
            #Try of the given cMap is really a cMap
            cList=pu.getColorList(len(timedWord)+1, cMap)
        else:
            #If not interpret it as a color
            cList=(len(timedWord)+1)*[cMap]
        
        cFun = self.funnelDict[funIn]
        C1_p = np.array(clocksIn[0]); C2_p = np.array(clocksIn[1])
        
        m = cFun.dynSys.stateDim
        #get current state
        dXc = XIn.reshape((m,1))
        Xc = dXc + self.funnelDict[funIn].traj.getX(guround(C1_p))
        #Plot initial state
        if ax!=None: pu.plotEllipse(ax[0], cFun.traj.getX(guround(C1_p))[dim,:], cFun.dynSys.getP()[np.ix_(dim,dim)] , cFun.lyapVal, color = cList[0], faceAlpha=0.5)
        if ax!=None: ax[0].plot(Xc[dim[0],:], Xc[dim[1],:], '.', color = cList[cNum], markersize=10.0, markerfacecolor = cList[cNum]) 
            
        #Set up return values
        success = True
        allXref = np.zeros((m,0))#Reference position
        allX = np.zeros((m,0))#Absolute position
        allDX = np.zeros((m,0))#Relative position
        allT = np.zeros((0,))#time array
        
        #Check initial invariants
        success = success and eval(cFun.invariantList)
        if not success:
                warnings.warn( 'Failed initial evaluation of invariants', UserWarning )
        
        #Simulate
        tLast = tGlobIn
        for aLetter in timedWord:
            #check if state in funnel
            success = success and cFun.locPinF(dXc)
            if not cFun.locPinF(dXc):
                warnings.warn('Run failed because current state outside current funnel.\nFunnel ID: {0}\nCurrent local position:{1}\ntime:{2}'.format( cFun.ID, str(dXc.squeeze()), tLast ), UserWarning)
            
            #Take care of the rounding
            thisT = guround( np.array( list(np.arange(0, aLetter[0], dT)) + [aLetter[0]] ) )#TBD uff ca fait moche #np.concatenate( (np.arange(0, aLetter[0], dT), np.array(aLetter[0]).squeeze()) )
            thisC1_p = guround(C1_p) + thisT
            thisC2_p = thisT
            thisDX = cFun.dynSys.propState(thisC2_p, dXc) # In real continuous time
            thisXref = cFun.traj.getX(thisC1_p)
            thisX = thisXref + thisDX
            #plot
            if ax!=None: ax[0].plot(thisXref[dim[0],:], thisXref[dim[1],:], color=cList[cNum])
            
            #Store the results
            allT = np.concatenate((allT, tLast+thisT))
            tLast = allT[-1]
            allXref = np.concatenate((allXref, thisXref), axis = 1)
            allX = np.concatenate((allX, thisX), axis = 1)
            allDX = np.concatenate((allDX, thisDX), axis = 1)
            
            
            #Calculate transition
            Xc = thisX[:,-1].reshape((m,1))
            if ax!=None: ax[0].plot(Xc[dim[0],:], Xc[dim[1],:], '.', color = cList[cNum], markersize=10.0, markerfacecolor = cList[cNum]) 
            #Update clocks
            C1_p+=aLetter[0] #rounded float time
            C2_p+=aLetter[0]
            
            if aLetter[1] != None:
                thisTrans = cFun.availableAtTo( C1_p, aLetter[1], aLetter[2], C2_p ) #availableAtTo is defined in transitionDict but can still be accessed via the subfunnel
                 
                #Perform transition
                if len(thisTrans)==0:
                    raise ValueError( 'No valid transition available for {0}'.format(str(aLetter)) )
                else:
                    if len(thisTrans)>1:
                        print('Multiple transitions available, choosing first')#Solved: arriving times has to be respected, so even if the transition is not unique, all in the list are equivalent
                    succesTrans, clockSet = thisTrans[0].doTrans(C1_p, C2_p, cFun, self.funnelDict[aLetter[1]], Xc)
                 
                #Do plotting if ax is given
                if ax != None: pu.plotTransition(ax[0], self, thisTrans[0], C1_p, C2_p, color=[cList[cNum], cList[cNum+1]], faceAlpha=faceAlpha)
                
                success = success and succesTrans
                if not succesTrans:
                    raise Warning('Transition from funnel {0} to {1} failed at time {2}'.format(cFun.ID, self.funnelDict[aLetter[1]].ID, aLetter[0]))
                #Set new clocks and funnel
                cFun = self.funnelDict[aLetter[1]]
                C1_p = clockSet[0]
                C2_p = clockSet[1]
            #Next color
            cNum+=1
            #Calculate the current error
            dXc = Xc-cFun.traj.getX(guround(C1_p))
                
        #Done 'simulation'
        #Plot the result
        if ax!=None:
            ax[0].plot( allX[dim[0],:], allX[dim[1],:], color=lineStyle[0], linestyle=lineStyle[1] )
            ax[1].plot( allT, allXref[oneDim,:], color=refStyle[0], linestyle=refStyle[1] )
            ax[1].plot( allT, allX[oneDim,:], color=lineStyle[0], linestyle=lineStyle[1]  )
            
            #ax[1].plot( allT, np.apply_along_axis(norm, axis=0, arr=allDX) )
        if tSampleIn != None:
            #Resample all the points
            allT, allXref, allX, allDX = doSampleInter( tSampleIn, allT, allXref, allX, allDX )
        
        return [success, allT, allXref, allX, allDX, [C1_p, C2_p]]
                
            
        
        
    
