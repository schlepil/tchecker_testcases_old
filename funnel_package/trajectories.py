## @package trajectories
# Implements classes that can be used as nominal trajectories for subfunnels ( @ref subFub ) 
# The trajectories evolve continuously. No discretization is performed. Only a warning is raised when the trajectory can not be properly discretized
# @todo Search for good spline implementation with a return of coefficients

from funnel_package.utils import *

## Base class from which all trajectories used in subfunnels (@ref subFun) should be derived or instanced.
#
# An analyticTraj describes a curve through the state space. The curve is described by the lambda function X0 
# and its derivative X0_dot (both taking the current time point as input). For convenience this curve can be
# run through at different speeds. This can be done by specifying a time dilatation factor 'alphaTime' different then one.
# @page analyticTraj 
class analyticTraj(object):
    ## Constructor
    #
    # @param mIn[int] Length of the state space vector
    # @param X0_In[lambda function] Analytic function taking a time point as input. Returns the corresponding point on the curve in the state-space
    # @param X0_dotIn[lambda function] Analytic function taking a time point as input. Returns the derivative of the curve at this point
    # @param tSpanIn[tuple of floats] Start and end time of the trajectory
    # @param isCyc[boolean] Flag indicating whether the trajectory is cyclic (so X0(tSpan[0])==X0(tSpan[1]) and X0_dot(tSpan[0])==X0_dot(tSpan[1])) or not. IMPORTANT: To ensure correct usage by the subfunnel (@ref subFun) and @ref funnel1 cyclic trajectories have to be marked as such.  
    # @param timeDilatFuncIn[lambda function] Function to keep the derivatives (in X0) consistent when alphaTime!=1.0. Has to return a (mIn,)-np.array and takes alphaTime (as float) as input.
    # @param timeDilatFuncDotIn[lambda function] Function to keep the derivatives (in X0_dot) consistent when alphaTime!=1.0. Has to return a (mIn,)-np.array and takes alphaTime (as float) as input.
    # @param mapUnittoTimeFuncIn[lambda function] Lambda function mapping a number (float in [0.0,1.0]) to a time point in [tSpan[0], tSpan[1]. This is used to smartly distribute the search points along the curve (see @ref getPossibleTrans_vC). Has to be in the form lambda p, tStart, tEnd: ... . 
    # @param N_searchIn[int] Number of search points distributed along the curve. Defaulted to 100.
    def __init__(self, mIn, X0In, X0_dotIn, tSpanIn, isCyc, timeDilatFuncIn, timeDilatFuncDotIn, mapUnittoTimeFuncIn = lambda p, tStart, tEnd: tEnd*p + tStart*(1-p), N_searchToIn=100, N_searchFromIn=None ):
        self.m = dp(mIn)
        self.X0 = dp(X0In)
        self.X0_dot = dp(X0_dotIn)
        self._tSpan = dp(np.array(tSpanIn).squeeze())
        assert self.tSpan[1] > self.tSpan[0], '''A trajectory is always defined for 'positive' tSpans''' #tSpan has only start and end time
        #Check if fitted for discretization
        if np.abs( self.tSpan[0]-gToNearest2(self.tSpan[0])) > 1.0e-2/globalRoundVal or np.abs( self.tSpan[1]-gToNearest2(self.tSpan[1])) > 1.0e-2/globalRoundVal2:
            warnings.warn('Trajectory with tSpan: {0} can not be exactly discretized!'.format( str(list(self.tSpan)) ), UserWarning )
         
        #If the trajectory is defined as being cyclic, the time tSpan[1]-tSpan[0] is assumed to be the cycle time
        self.isCycFlag = dp(isCyc)
        
        #Set a time dilatation factor
        self._alphaTime = 1.0;#To initialize
        self.tDF = dp(timeDilatFuncIn)
        self.tDFc = np.array(self.tDF(self.alphaTime)).squeeze()#reshape((self.m,1))
        self.tDFdot = dp(timeDilatFuncDotIn)
        self.tDFdotc = np.array(self.tDFdot(self.alphaTime)).squeeze()#reshape((self.m,1))
        
        #Work around to avoid point dependent alpha #TBD recode properly with point dependent alpha
        #Define a function that maps the interval [0,1] to tSpan. this not necessarily a linear function
        #This function has to be set explicitly if chosen non trivial
        self._mapUnittoTimeFunc_2 = None
        self._mapUnittoTimeFunc = None
        self.mapUnittoTimeFunc = mapUnittoTimeFuncIn
        
        self.N_searchTo = N_searchToIn
        self.N_searchFrom = N_searchFromIn if N_searchFromIn != None else 4*self.N_searchTo
        self.adjustSearchPoints=True
        
        self.alphaTime = 1.0;#to calculate all needed data
        
    ## Access to the elementwise biggest absolute values found in X0_dot along the trajectory
    # @return [(m,1)-np.array] Biggest absolute values of the derivatives along the trajectory
    @property
    def X0_dot_absMax(self):
        return self._X0_dot_absMax
    
    ## Returns the tSpan
    # @return [tuple of floats] Returns the (start, end) - time
    @property
    def tSpan(self):
        return self._tSpan
    ## Sets a new tSpan. This should normally not be necessary
    @tSpan.setter
    def tSpan(self, tSpanIn):
        warnings.warn('Changing tSpan manually, this should be done with great precaution!', UserWarning)
        self._tSpan = tSpanIn
    
    ## decorator for mapUnittoTimeFunc setter. mapUnittoTimeFunc_2 is directly created. It is based on the same lambda function but already takes into account the current tSpan 
    # @param lambdaFunc[lambda function] Has to be of the form func = lambda p, tStart, tEnd: 'Your mapper function'    
    @property
    def mapUnittoTimeFunc(self):
        return self._mapUnittoTimeFunc
    @mapUnittoTimeFunc.setter
    def mapUnittoTimeFunc(self, lambdaFunc):
        self._mapUnittoTimeFunc = lambdaFunc
        self._mapUnittoTimeFunc_2 = lambda p: self.mapUnittoTimeFunc(p, self.tSpan[0], self.tSpan[1])  
    @property
    def mapUnittoTimeFunc_2(self):
        return self._mapUnittoTimeFunc_2
    
    ## Uses the mapUnittoTimeFunc to calculate the time points corresponding to the lambdapoints
    # @param lambdaPoints[(n,)-np.array] Vector of floats in [0,1] to be mapped into tSpan
    # return [(n,)-np.array] With the corresponding time points 
    def unitToTime(self, lambdaPoints):
        assert (np.all(lambdaPoints>=0.0) and np.all(lambdaPoints<=1.0)), 'Lambdapoints exceed [0.0,1.0]'
        lambdaPoints = np.array(lambdaPoints).squeeze(); lambdaPoints = lambdaPoints.reshape((lambdaPoints.size,))
        return np.apply_along_axis( self.mapUnittoTimeFunc_2, 0, lambdaPoints )
        #Wrap if cyclic #No
#         if self.isCycFlag:
#             lambdaPoints = np.remainder(lambdaPoints, 1.0)
#         else:
#             assert (np.all(lambdaPoints>=0.0) and np.all(lambdaPoints<=1.0)), 'Lambdapoints exceed [0.0,1.0]'
        

    ## calculates the points on the curve in the state space corresponding to given time points
    # @param t[float or (n,)-np.array] Time points for which the points in the state space shall be calculated. If the trajectory is NOT cyclic than the calculation will fail when any t is outside the tSpan. If the trajectory IS cyclic, then the calculation will be performed on the 'wrapped time'. 
    # @return [(m,n)-np.array] Matrix where each column is a point in the m-dimensional state-space
    def getX(self, t):
        t = np.array(t);t=t.reshape((t.size,))
        t = self.checkT(t)
        tS = t/self.alphaTime #Evaluate at the 'dilated' time point 
        resX = np.zeros((self.m, tS.size))
        for k in range(0,tS.size):
            resX[:,k] = mult(np.array(self.X0(tS[k])).squeeze(), self.tDFc).squeeze()#Relax the derivativees
        return resX
    ## calculates the points on the curve in the state space corresponding to given lambdapoints
    # @param lambdapoints[float or (n,)-np.array] Lambdapoints for which the points in the state space shall be calculated. If the trajectory is NOT cyclic than the calculation will fail when any t is outside [0,1]. If the trajectory IS cyclic, then the calculation will be performed on the 'wrapped points'. 
    # @return [(m,n)-np.array] Matrix where each column is a point in the m-dimensional state-space
    def getX2(self, lambdapoints):
        return(self.getX(self.unitToTime(lambdapoints)))
    
    ## calculates the derivatives of the curve in the state space corresponding to given time points
    # @param t[float or (n,)-np.array] Time points for which the derivatives in the state space shall be calculated. If the trajectory is NOT cyclic than the calculation will fail when any t is outside the tSpan. If the trajectory IS cyclic, then the calculation will be performed on the 'wrapped time'. 
    # @return [(m,n)-np.array] Matrix where each column is the vector of derivatives in the m-dimensional state-space
    def getXdot(self, t):
        t = np.array(t); t=t.reshape((t.size,))
        t = self.checkT(t)
        tS = t/self.alphaTime #Evaluate at the 'dilated' time point
        resX = np.zeros((self.m, tS.size))
        for k in range(0,tS.size):
            resX[:,k] = mult(np.array(self.X0_dot(tS[k])).squeeze(), self.tDFdotc).squeeze()#Relax the derivatives        
        return resX
    ## calculates the derivatives of the curve in the state space corresponding to given lambdapoints
    # @param t[float or (n,)-np.array] Lambdapoints for which the derivatives in the state space shall be calculated. If the trajectory is NOT cyclic than the calculation will fail when any t is outside the tSpan. If the trajectory IS cyclic, then the calculation will be performed on the 'wrapped points'. 
    # @return [(m,n)-np.array] Matrix where each column is the vector of derivatives in the m-dimensional state-space
    def getXdot2(self, lambdapoints):
        return(self.getXdot(self.unitToTime(lambdapoints)))
    
    ## returns a vector holding the (absolute) maximum of each element along the trajectory
    # @param atSpan[tuple of floats or None] Specifies the start and end of the section of the curve to be considered. If None (default) than the whole trajectory is searched. 
    # @return [m,1]-Matrix Containing the largest absolute values for each element found
    def getMaxXdot(self, atSpan=None):
        ###Return the maximum of absolute(X0_dot)
        if atSpan is None:
            return self.X0_dot_absMax
        allT = self.unitToTime(np.linspace(0, 1, self.N_searchFrom))
        ind = np.flatnonzero( np.logical_and( allT >= atSpan[0], allT <= atSpan[1] ) )
        allT = allT[ind]
        allXdot = self.getXdot(allT)
        return np.amax( nabs(allXdot), 1 ).reshape((self.m,1))#TBD: This might be to conservative consider checking every time step with 
    ## returns a vector holding the (absolute) maximum of each element along the trajectory
    # @param aLambdaSpan[tuple of floats or None] Specifies the start and end of the section of the curve to be considered. If None (default) than the whole trajectory is searched. 
    # @return [m,1]-Matrix Containing the largest absolute values for each element found
    def getMaxXdot2(self, aLambdaSpan=None):
        ###Return the maximum of absolute(X0_dot)
        if aLambdaSpan is None:
            return self.X0_dot_absMax
        allL = np.concatenate( ( self.unitToTime(np.arange(0, 1, self.N_searchFrom)), np.array([aLambdaSpan[1]]) ) )
        allXdot = self.getXdot2(allL)
        return np.amax( nabs(allXdot), 1 ).reshape((self.m,1))#TBD: This might be to conservative consider checking every time step with 
    
    ## Helper function to ensure time consistency
    def checkT(self, t):
        #Check if time has to be wrapped (cyclic trajectory) or if it exceeds the limits of the traj
        #No cyclic wrapping. A transition has to be taken and values always have to stay in the interval
        t = np.array(t)
        try:
            assert (np.all(t>=self.tSpan[0]-(1e-2/globalRoundVal)) and np.all(t<=self.tSpan[1]+(1e-2/globalRoundVal))), 'Time exceeds funnel boundaries'
        except:
            assert (np.all(t>=self.tSpan[0]-(1e-2/globalRoundVal)) and np.all(t<=self.tSpan[1]+(1e-2/globalRoundVal))), 'Time exceeds funnel boundaries'
#         if self.isCycFlag:
#             t2 = t - self.tSpan[0]
#             tr = np.remainder(t2, self.tSpan[1]-self.tSpan[0])
#             t = self.tSpan[0]+tr
#         if not self.isCycFlag:
#             assert (np.all(t>=self.tSpan[0]) and np.all(t<=self.tSpan[1])), 'Time exceeds funnel boundaries'
        return t
    
    @property
    def alphaTime(self):
        return self._alphaTime
    @alphaTime.setter
    def alphaTime(self, alphaIn):
        if alphaIn == 0: raise ZeroDivisionError('Time dilatation factor alpha needs to be non-zero')
        
        self._tSpan = self.tSpan*alphaIn/self.alphaTime
        #Adjust search point density
        if self.adjustSearchPoints:
            self.N_searchTo = int(self.N_searchTo*alphaIn/self.alphaTime)
            self.N_searchFrom = int(self.N_searchFrom*alphaIn/self.alphaTime)
        
        if np.abs( self.tSpan[0]-gToNearest(self.tSpan[0])) > 5.0e-1/globalRoundVal or np.abs( self.tSpan[1]-gToNearest(self.tSpan[1])) > 5.0e-1/globalRoundVal2:
            warnings.warn('Trajectory with tSpan: {0} can not be exactly discretized!'.format( str(list(self.tSpan)) ), UserWarning )
        self._tSpan = gToNearest(self.tSpan)
        #TBD: extra treatment if alphaTime turns negative
        self.tDFc = np.array(self.tDF(alphaIn)).squeeze()
        self.tDFdotc = np.array(self.tDFdot(alphaIn)).squeeze()
        
        self._X0_dot_absMax = self.getMaxXdot2([0.0,1.0])
        self._alphaTime = alphaIn
        
        #Reset the mapping function
        self._mapUnittoTimeFunc_2 = lambda p: self.mapUnittoTimeFunc(p, self.tSpan[0], self.tSpan[1])
        return 0
    
## convinience class to quickly generate linear trajectories
class linTraj(analyticTraj):
    ## Constructor
    #
    # Automatically generates all necessary functions based on defList
    # @param defList[list of lists] Has to contain a sublist for each dimension. The sublist have to be of the form (gamma, K) where gamma is the velocity in this direction and K is number of derivatives of this directions in the state space
    # @param tSpanIn[tuple of floats] See @ref analyticTraj
    # @param N_searchIn[integer] See @ref analyticTraj
    def __init__(self, defList, tSpanIn, N_searchFromIn=100, N_searchToIn=None):
        ###Creates an analytical straight line trajectory
        #defList=( dim1=(alpha, nbr od derivs), ... )
        #Resulting trajectory has the form [x1, x1_dot, x1_ddot, x2,...]' 
        coeffsVel = []
        coeffsPos = []
        m=0
        #Also assemble the time dilatation function
        fTD = 'lambda alpha: ['
        fTDdot = 'lambda alpha: ['
        for aDim in defList:
            for k in range(aDim[1]):
                m+=1
                fTD = fTD + '1.0/alpha**{0:d},'.format(k)
                fTDdot = fTDdot + '1.0/alpha**{0:d},'.format(k+1)
                coeffsPos.append( aDim[0] if k == 0 else 0 )
                coeffsVel.append( aDim[0] if k == 1 else 0 )
        #Format string and evaluate
        fTD=fTD[0:-1]+']'
        fTD = eval(fTD)
        fTD2 = lambda alpha: np.array(fTD(alpha)).reshape((m,1))
        fTDdot=fTDdot[0:-1]+']'
        fTDdot = eval(fTDdot)
        fTDdot2 = lambda alpha: np.array(fTDdot(alpha)).reshape((m,1))
        #Transform into numpy
        coeffsPos = np.array(coeffsPos).reshape((m,1))
        coeffsVel = np.array(coeffsVel).reshape((m,1))
        #Create lambda functions
        X0 = lambda t: coeffsPos*t + coeffsVel
        X0_dot = lambda t: coeffsVel
        
        super(linTraj, self).__init__(m, X0, X0_dot, tSpanIn, False, fTD2, fTDdot2, N_searchFromIn=N_searchFromIn, N_searchToIn=N_searchToIn)
#####################