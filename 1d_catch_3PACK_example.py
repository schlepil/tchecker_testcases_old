from funnel_package import *

import os

# MODIFIERS
# basic
searchPointFact = 2. # The higher the more transitions are available; Default = 2
convertTime = 3 #Multiplier for discrete time-steps per seconds; Default value 1 corresponds to 10000steps per sec
# funnel system
# L'atteignabilité est en ce moment assez sensible sur ces parametres en ce moment
# Changez plutot les autres
numVelFun = 4# Number of different velocities; Default was 4
numSizeFun = 5# Number of sub-sizes; Default was 5

# Task
# One can change when, where and how many packages have to arrive
maxCarry = "2" # Maximal number of items carried at a time
packageGlobTime = [50.*globalRoundVal, 100.*globalRoundVal, 150.*globalRoundVal] # Arrival time of packages
packageLanes = [3,2,1] # Lanes where they arrive
deltaTime = 0.025 # Maximally allowed advance/delay for pick up; The smaller, the harder the task
                  #]0, 0.05]


# END MODIFIERS
assert len(packageLanes) == len(packageGlobTime)

# Discretization
up = lambda x: int(np.ceil(x * convertTime))
down = lambda x: int(np.floor(x * convertTime))
mid = lambda x: int(x * convertTime)


uQR = QRstruct

aD = lambda vDiag: np.diag(vDiag)
a1 = lambda V: np.array(V).reshape((1,1))
r2 = sqrt(2)

#Only one 1d ref trajectory with a second order system
#Create the reference as analytical lambda Function
vMax=1.0
maxDist = 15.0
tSpan=[0.0,maxDist/vMax]
f_ref_f = lambda t: [vMax*t, vMax]
fdot_ref_f = lambda t: [vMax, 0.0]
f_ref_b = lambda t: [maxDist-vMax*t, -vMax]
fdot_ref_b = lambda t: [-vMax, 0.0]

#Create time dilatation functions
tDF = lambda alphaT: [1.0, 1.0/alphaT]
tDFdot = lambda alphaT: [1.0/alphaT, 1.0/alphaT**2.0]

#m=2; isCyc=False; N_searchFrom = numbLanes*100; N_searchTo = numbLanes*20
m=2
isCyc=False
N_searchFrom = int(maxDist/2.*15*searchPointFact)
N_searchTo = int(maxDist*3.0/3.*0.5*searchPointFact)
dLf1=0.001
doAdjust = True

myTraj_f = traj.analyticTraj(m, f_ref_f, fdot_ref_f, tSpan, isCyc, tDF, tDFdot, N_searchFromIn=N_searchFrom, N_searchToIn=N_searchTo)
myTraj_b = traj.analyticTraj(m, f_ref_b, fdot_ref_b, tSpan, isCyc, tDF, tDFdot, N_searchFromIn=N_searchFrom, N_searchToIn=N_searchTo)

myTraj_f.adjustSearchPoints = doAdjust
myTraj_b.adjustSearchPoints = doAdjust

#create a dynamical system
A = np.array([[0.0,1.0], [0.0, -.1]])#free mass with small viscous damping
B = np.array([0.0, 1.0/5.0]).reshape((2,1)) #Weighing five kilos and being directly actuated by a force

XdotLim = np.array([1.4*vMax,999999.0]).reshape((m,1))#Constraints on acceleration and not force/input
mySys = msp.linSys(A, B, XdotLim)
XdotLimStat = np.array([0.35*vMax,999999.0]).reshape((m,1))#Constraints on acceleration and not force/input
mySysStat = msp.linSys(A, B, XdotLimStat)

#Put it together into a funnel#
QRlist= uQR(aD([1.0,1.0]),a1(2.0))
QRstat= uQR(aD([1.0,1.0]),a1(2.0))

numStat = 4
dX = maxDist/(numStat-1)

contracCoeff1 = 0.4
targetAlphasIn1 = np.linspace(0,1,numVelFun)**2*5+1 if numVelFun != 4 else np.array([1.0, 1.25, 1.9, 6]) #Attention smaller number correspond to higher speeds (Directly scales the time needed to travel to whole funnel length!

myFunn_f = fc.funnel1(myTraj_f, mySys, QRlist, targetAlphasIn=targetAlphasIn1, numVelFun=numVelFun, numSizeFun=numSizeFun, contracCoeff=contracCoeff1, dLfIn=dLf1, tScale=convertTime)#convertTime/globalRoundVal)
myFunn_b = fc.funnel1(myTraj_b, mySys, QRlist, targetAlphasIn=targetAlphasIn1, numVelFun=numVelFun, numSizeFun=numSizeFun, contracCoeff=contracCoeff1, dLfIn=dLf1, tScale=convertTime)#convertTime/globalRoundVal)
myFunn_stat = []
for k in range(numStat):
    myFunn_stat.append( fc.funnel1(traj.analyticTraj(m, eval('lambda t:[(dX*{0}),0.0]'.format(float(k))), lambda t:[0.0,0.0], [0.0,1000.0], True, tDF, tDFdot, N_searchFromIn=1, N_searchToIn=1), mySysStat, QRstat, numVelFun=1, numSizeFun=3, contracCoeff=0.5**2.0, dLfIn=0.2, tScale=convertTime))#convertTime/globalRoundVal) )
# Save the ID of the starting funnel -> Here we want to start in the deposit lane in the smallest funnel

myTimedAut = fc.timedAutomata()
myTimedAut.addFunnel(myFunn_f)
myTimedAut.addFunnel(myFunn_b)
for funn in myFunn_stat:
    myTimedAut.addFunnel(funn)
myTimedAut.uniqueID()
myTimedAut.calcTransition(tScale=convertTime)#(tScale=convertTime/globalRoundVal)
myTimedAut.createInvariants()

initDef = { "__initFunnel__":myFunn_stat[0].funnelSys[0][-1].ID, "__initCtrlClock__":str(mid(myFunn_stat[0].funnelSys[0][-1].tSpan[0])), "__initLocalClock__":"0"}
initDef['__maxCarry__'] = maxCarry
initDef['__nbrPack__'] = f"{len(packageLanes):d}"

# Manually add the catch and deliver transitions
# Dynamics funnels -> can be caught if in small and slow
# Forward trajectory
# Slow
for aFunnelVelSys in myFunn_f.funnelSys[(numVelFun*2)//3:]:
    # Small
    for aSubFunnel in aFunnelVelSys[(numSizeFun*2)//3:]:
        #Here we can add the catching transitions
        for aLane in range(1,4):
            #transStruct(self, parentIDIn, childIDIn, tSpanIn, transGuardListIn, transSetListIn, evalStrIn='{0:d}', plotStrIn='{0:.3f}')
            ctrlTime = aSubFunnel.tSpan[1]*aLane/4
            thisTSpan = [ctrlTime,ctrlTime]
            thisGuardList = [["==", "C1_p", mid(ctrlTime)]]
            thisSetList = [] # All clocks and variables keep their values
            thisTrans = transStruct(aSubFunnel.ID, aSubFunnel.ID, thisTSpan, thisGuardList, thisSetList)
            thisTrans.event = "catch"
            aSubFunnel.transDict.add( thisTrans )
        # Delivering transition
        ctrlTime = 0
        thisTSpan = [ctrlTime, ctrlTime]
        thisGuardList = [["==", "C1_p", mid(ctrlTime)]]
        thisSetList = []  # All clocks and variables keep their values
        thisTrans = transStruct(aSubFunnel.ID, aSubFunnel.ID, thisTSpan, thisGuardList, thisSetList)
        thisTrans.event = "deliver"
        aSubFunnel.transDict.add(thisTrans)

# Backward trajectory
# Slow
for aFunnelVelSys in myFunn_b.funnelSys[(numVelFun*2)//3:]:
    # Small
    for aSubFunnel in aFunnelVelSys[(numSizeFun*2)//3:]:
        #Here we can add the catching transitions
        for aLane in range(1,4):
            #transStruct(self, parentIDIn, childIDIn, tSpanIn, transGuardListIn, transSetListIn, evalStrIn='{0:d}', plotStrIn='{0:.3f}')
            ctrlTime = max(aSubFunnel.tSpan[1] - aSubFunnel.tSpan[1]*aLane/4,0)
            thisTSpan = [ctrlTime,ctrlTime]
            thisGuardList = [["==", "C1_p", mid(ctrlTime*globalRoundVal)]]
            thisSetList = [] # All clocks and variables keep their values
            thisTrans = transStruct(aSubFunnel.ID, aSubFunnel.ID, thisTSpan, thisGuardList, thisSetList)
            thisTrans.event = "catch"
            aSubFunnel.transDict.add( thisTrans )
        # Delivering transition
        ctrlTime = 0
        thisTSpan = [ctrlTime, ctrlTime]
        thisGuardList = [["==", "C1_p", mid(ctrlTime*globalRoundVal)]]
        thisSetList = []  # All clocks and variables keep their values
        thisTrans = transStruct(aSubFunnel.ID, aSubFunnel.ID, thisTSpan, thisGuardList, thisSetList)
        thisTrans.event = "deliver"
        aSubFunnel.transDict.add(thisTrans)

# Converging / static trajectories
for aStatFunnel in myFunn_stat:
    aSubFunnel = aStatFunnel.funnelSys[0][-1] # Allow to deliver only for smallest

    thisTSpan = [mid(aSubFunnel.tSpan[0]), mid(aSubFunnel.tSpan[1])]
    thisGuardList = [] # It is always possible to deliver
    thisSetList = []
    thisTrans = transStruct(aSubFunnel.ID, aSubFunnel.ID, thisTSpan, thisGuardList, thisSetList)
    thisTrans.event = "deliver"
    aSubFunnel.transDict.add(thisTrans)

# Construct belt transition string
# À la main...
#['<=', max(down(packageGlobTime-deltaTime),0), "C0_p"], ['<=', "C0_p", up(packageGlobTime+deltaTime)]
beltArrivingTransition = ""
for aGlobTime in packageGlobTime:
    beltArrivingTransition += f"edge:belt:approach:arrived:noaction{{provided:glob>={max(down(aGlobTime-deltaTime),0)} && glob<={up(aGlobTime+deltaTime)}}}{os.linesep}"

initDef["__BeltArrivingTrans__"] = beltArrivingTransition

#Get all system | robot transitions
robLocList, robTransList = tcheckUtils.TAToTcheckerEdges(myTimedAut, procName="rob", ctrlClkStr="ctrl", localClkStr="local")


tempStr = ""
for aLoc in robLocList:
    tempStr += aLoc
initDef["__RobFunnelStates__"] = tempStr

tempStr = ""
for aTrans in robTransList:
    tempStr += aTrans
initDef["__RobFunnelTrans__"] = tempStr

# Replace all
with open("./tchecker_files/templates/1d_catch_states_template_2.tcheck", "r") as templ, open(f"./tchecker_files/problems/1d_catch_states_2_{numVelFun}_{numSizeFun}_{searchPointFact}_{convertTime}.tcheck", "w") as prob:

    for aLine in templ:
        for aKey,aVal in initDef.items():
            aLine = aLine.replace(aKey, aVal)
        print(aLine)
        prob.write(aLine)

#Done








