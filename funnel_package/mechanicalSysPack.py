###Package used to simplify the usage of dynamical systems (plant)
#At the moment only LQR is implemented
import control as ctrl

from funnel_package.utils import *

#####################
class dynSys(object):
    ##abstract class
    def __init__(self, nIn, mIn, X0In=None, QIn=None, RIn=None):
        self.stateDim = nIn
        self.inputDim = mIn
        
        if QIn!=None: assert np.min((QIn.shape[0] == QIn.shape[1] and QIn.shape[0]==self.stateDim)), 'Q matrix for lqr needs to be symmetric positive definite and of the same size as A'
        if RIn!=None: assert np.min((RIn.shape[0]==RIn.shape[1] and RIn.shape[0]==self.inputDim)), 'R matrix for lqr needs to be symmetric positive semidefinite and of the same size as the dolumn space of B'
        
        self.Q = QIn
        self.R = RIn
        self.P = None
        self.Pc = None
        self.Abar = None
        self.Bbar = None
        
        self.lqrFlag = False
        #self.calcCtrl()
        
        if X0In is None: X0In=np.zeros((self.stateDim,1))
        self.X0 = np.array(X0In).reshape((self.stateDim,1))
        self.Xc = self.X0 
    
    def getA(self):
        ###Function to return A matrix from x_dot = A*x + B*u
        raise NotImplementedError
    
    def getB(self):
        ###Function to return A matrix from x_dot = A*x + B*u
        raise NotImplementedError
    
    def getAbar(self):
        if not self.lqrFlag:
            self.calcCtrl()
        return self.Abar
    
    def getBbar(self):
        if not self.lqrFlag:
            self.calcCtrl()
        return self.Bbar
    
    def setQ(self, QIn):
        self.Q = QIn
        self.Abar = None
        self.Bbar = None
        return self.calcCtrl()
    
    def setR(self, RIn):
        self.R = RIn
        
        self.Abar = None
        self.Bbar = None
        return self.calcCtrl()
    
    def getP(self):
        if not self.lqrFlag:
            self.calcCtrl()
        return self.P
    
    def getPc(self):
        if not self.lqrFlag:
            self.calcCtrl()
        return self.Pc
    
    def calcCtrl(self):
        #Apply the LQR control if possible
        if self.Q is None or self.R is None:
            self.P = None
            self.Abar = None
            return 1
        K,P,_unused = ctrl.lqr(self.A, self.B, self.Q, self.R)
        self.P = P
        self.Pc=chol(P)
        self.Bbar = dot(self.B, K)
        self.Abar = self.A - self.Bbar
        self.lqrFlag = True
        return 0
    
    def getLyapDot(self):
        raise NotImplementedError

    def getXdotLim(self, aX, *args):
        ###Get the maximal Xdot the system can deliver around aX
        raise NotImplementedError
        
#####################
class linSys(dynSys):
    ###Implements a LTI-system for the use in a Funnel automata
    def __init__(self, AIn, BIn, X_dotLimIn, X0In=None, QIn=None, RIn=None):
        
        #check
        assert AIn.shape[0]==AIn.shape[1], 'A matrix for the system xdot = Ax+Bu needs to be square'
        assert BIn.shape[0]==AIn.shape[0], 'B matrix for the system xdot = Ax+Bu needs to have the same number of rows as A'
        
        self.A = AIn
        self.B = BIn
        
        super(linSys, self).__init__(AIn.shape[0], BIn.shape[1], X0In, QIn, RIn)
        
        self.X_dotLim = np.fabs( np.array(X_dotLimIn).reshape((self.stateDim,1)) )
        
    def getA(self):
        return self.A
    
    def getB(self):
        return self.B
    
    def propState(self, tEnd, X0In=None, doStore=False ):
        ###Evolve state using matrix exponential
        #!TBD: Check the different expm implementations 
        X = X0In if X0In!= None else self.Xc
        
        #X = dot(expm(self.getAbar()*tEnd), X)
        #if doStore: self.Xc = X
        Xout = np.zeros((self.stateDim,tEnd.size))
        X = X.reshape((self.stateDim,1))
        for k in range(tEnd.size):
            Xout[:,k] = dot(expm(self.getAbar()*tEnd[k]), X).squeeze()
             
        if doStore: self.Xc = X[:,-1].reshape((self.stateDim,1))
        return Xout
    
    def getXdotLim(self, aX, *args):
        ###Get the maximal Xdot the system can deliver around the point aX
        return self.X_dotLim
    def getLyapDot(self, *_unused):
        #return d/dt V(X)
        if not self.lqrFlag:
            self.calcCtrl()
        if self.lqrFlag:
            return -self.Q - ndot(self.P, self.B, inv(self.R), self.B.T, self.P)
        else:
            raise AttributeError('getLyapDot was called on linear systems before QR were defined!')
#####################
#Implements a 'perfectly' controlled system (so no feedback)
class ctrlSys(dynSys):
    def __init__(self, AbarIn, X_dotLimIn, X0In=None, QIn=None):
        #Creates a perfectly controlled system without feedback of the form
        #xDot = Abar*x
        #Abar: Hurwitz stable matrix
        #Q: positive DEFINITE matrix
        #!No checkup is performed on entry
        
        super(ctrlSys, self).__init__(AbarIn.shape[0], 0, X0In, QIn, None)
        
        self.Abar = AbarIn
        self.B = np.zeros((1,1))
        self.X_dotLim = np.fabs( np.array(X_dotLimIn).reshape((self.stateDim,1)) )
    
    def setR(self, RIn=None):
        pass#Dummy function
    
    def getBbar(self):
        raise AttributeError('Does not existed for controlled systems')
    
    def setQ(self, QIn):
        #In linSys control.lqr is used and it solves the problem P*A+A.T*P-P*B*inv(R)*B.T*P+Q=0 so Q needs to be positive definite
        #Here sp.linalg.lyap_solve and used and the problem has the from A*P+P*A.T=Q => multiply Q with minus one
        self.Q = QIn
        self.P = None
        return self.calcCtrl()
    
    def getP(self):
        if not self.lqrFlag:
            self.calcCtrl()
        return self.P
    
    def calcCtrl(self):
        #Apply the LQR control if possible
        if self.Q is None:
            self.P = None
            return 1
        P = solve_lyapunov(self.Abar.T, -1.0*self.Q)#Multiplication with -1.0 necessary due to the structure of the underlying solver Attention lyap_solve solves the dual problem(can this be said like that?)No need to transpose Q since its symmetric 
        #P=ctrl.lqr(self.Abar, np.zeros((1,1)), self.Q, np.zeros((1,1)))
        self.P = P
        self.Pc = chol(self.P)
        self.lqrFlag = True
        return 0
    
    def getXdotLim(self, aX, *args):
        ###Get the maximal Xdot the system can deliver around the point aX
        return self.X_dotLim
    
    def getLyapDot(self):
        if not self.lqrFlag:
            self.calcCtrl()
        if self.lqrFlag:
            return -self.Q
        else:
            raise AttributeError('getLyapDot was called on controlled systems before Q was defined!')
        
    def propState(self, tEnd, X0In=None, doStore=False ):
        ###Evolve state using matrix exponential
        #!TBD: Check the different expm implementations 
        X = X0In if X0In!= None else self.Xc
        
        tEnd = np.array(tEnd).squeeze()
        Xout = np.zeros((self.stateDim,tEnd.size))
        X = X.reshape((self.stateDim,1))
        for k in range(tEnd.size):
            Xout[:,k] = dot(expm(self.getAbar()*tEnd[k]), X).squeeze()
             
        if doStore: self.Xc = X[:,-1].reshape((self.stateDim,1))
        return Xout
