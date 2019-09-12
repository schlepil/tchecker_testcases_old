## @package otherUtils
#
# Implements useful functions and classes for which i do not have a better place yet
from funnel_package.utils import *

from funnel_package import funnelClasses as fc
import funnel_package.plotUtils as pu
## @class multiAutomat
#
# Class used for product automata. These automata are not really generated here, but within uppaal.
# This class is currently only simulating multiple timed words
class multiAutomat(object):
    
    ## Constructor
    #
    # Initializing an instance of the product automata. The @ref timedAutomata can either be directly given or added later on.
    # They will be stored in a list.
    # @param *args [@ref timedAutomata] Any number of timed automatas
    def __init__(self, *args):
        ## Ordered list of automats
        self._automatList = []
        self.size = 0
        for aAuto in args:
            self.addAutomat(aAuto)
    
    @property
    def automatList(self):
        return self.automatList
    @automatList.deleter
    def automatList(self):
        self._automatList = []
    
    ## Adds a timedAutomat (see @ref timedAutomata)  to the productAutomat
    #
    # @param aAuto [@ref timedAutomata] Automat to be added
    def addAutomat(self, aAuto):
        assert isinstance(aAuto, fc.timedAutomata), 'Only timedAutomata can be added to a productAutomat!'
        self._automatList.append(aAuto)
        self.size+=1
        return 0
    
    ## Function used to simulation a list of timed words describing the evolution of the states in the product automat.
    #
    # @param *List All the parameters containing 'List' in the name have to be a list of the corresponding parameters of @ref timedAutomata.simulate
    # @param other All other parameters are always passed to @ref timedAutomata.simulate
    # @return List List of the return values of @ref timedAutomata.simulate 
    def simulate(self, XInList, funInList, clocksInList, timedWordList, dT, ax = None, cMap='gnuplot2', faceAlpha=0.5, dim=[0,1], oneDim=0, lineStyleList=None, refStyleList=None):
        
        assert self.size==len(timedWordList), 'The number of automatas and timed words do not match'
        #Make new lists of timed words all having the same length
        newTimedWordList = [[] for k in range(len(timedWordList))]
        
        check = False
        
        if False:
            #Modify code so that it takes into account the new clock sets and the input clocks
            while True:
                check=True
                #get next transition
                tmin = np.Inf
                for k in range(self.size):
                    if len(timedWordList[k])!=0:
                        check = False
                        tmin = min(tmin, timedWordList[k][0][0]) #Time till next transition
                #Modify and append
                if check == True:
                    break
                for k in range(self.size):
                    if len(timedWordList[k])==0:
                        #Add dummy transition
                        newTimedWordList[k].append( (tmin, None) )
                    elif len(timedWordList[k])!=0 and tmin != timedWordList[k][0][0]:
                        #Add dummy transition and modify time to real transition
                        newTimedWordList[k].append( (tmin, None) )
                        timedWordList[k][0][0] = timedWordList[k][0][0]-tmin
                    else:
                        #This is the next transition -> append to new and delete from old
                        newTimedWordList[k].append(timedWordList[k].pop(0))
                    
        newTimedWordList = timedWordList
        nTW = len(newTimedWordList[0])
        #New wordList created. Create result list and perform simulation
        
        #But get some colors first
        cNum=0
        cList=pu.getColorList(len(timedWordList), cMap)
        
        allTList =[]; allXrefList = []; allXList = []; allDXList = []; successList = []
        
        for k in range(self.size):
            aSuccess, aAllT, aAllXref, aAllX, aAllDX, [aC1_p, aC2_p] = self._automatList[k].unguarded_simulate(XInList[k], funInList[k], clocksInList[k], newTimedWordList[k], dT, ax = ax, cMap=cList[k], faceAlpha=faceAlpha, dim=dim, oneDim=oneDim, lineStyle=[cList[k], '-'], refStyle=[cList[k], '--'])
            allTList.append(aAllT);
            allXrefList.append(aAllXref);
            allXList.append(aAllX);
            allDXList.append(aAllDX);
            successList.append(aSuccess);
        return successList, allTList, allXrefList, allXList, allDXList
        
#         allTList =[]
#         allXrefList = []
#         allXList = []
#         allDXList = []
#         successList = []
#         tLast=0
#         for k in range(len(newTimedWordList)):
#             s = self.automatList.funnelDict['a0'].shape #Funnel 'a0' is the first to be created within a timedAutomat. If this does not exist, than there are no funnels...
#             allTList.append(np.zeros(0,))
#             allXrefList.append(np.zeros(s))
#             allXList.append(np.zeros(s))
#             allDXList.append(np.zeros(s))
#             successList.append(True)
#         
#         #Do the simulation
#         for k in range(nTW):
#             for l in range(self.size):
#                 success, allT, allXref, allX, allDX, [C1_p, C2_p] = self.automatList[l].simulate(XIn, funIn, clocksIn, timedWord, dT, ax = None, cMap='jet', faceAlpha=0.5, dim=[0,1], oneDim=0, lineStyle='k', refStyle='--r'):
        
        
                
                
        
        
        
        
