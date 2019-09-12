## @package Package holding usefull function to interface python and uppaal

from funnel_package.utils import *
import re

##Loops through a real trace and returns timed words that can be simulated 
def traceToTW(tDict, aFile, xmlFile, fact, stateDict=None, initState='_discard_waiting', mode = 'unguarded'):
    
    #Get all the definitions in the xmlFile
    #It is assumed that the guards on the transitions are CONSTANT, e.g. they do NOT depend on the current state of the variables
    #Declarations should (must?) with a comment
    #Declarations in the xml file need to be of the form 'const int abc = 95858...528;'
    declarationFunc = re.compile("const int ([\w *+-/=]+);")
    getDeclaration = lambda aLine: declarationFunc.findall(aLine)[0];
    decDict={}
    for aLine in xmlFile:
        try:
            aDec = getDeclaration(aLine)
        except:
            continue
        aDec = aDec.replace(' ','') #Apply changement
        #exec(aDec)
        aDec = aDec.split('=')
        assert len(aDec)==2, 'unrecognised declaration in:\n'+aLine
        decDict[aDec[0]] = eval(aDec[1], {}, decDict) #Create local variable
        
    #getC1Func = re.compile("C1 := (\d+)[\, , }]")
    getC1Func = re.compile("C1 := ([\w *+-/]+)[,}]")
    getC1=lambda aLine: float(eval(getC1Func.findall(aLine)[0], {}, decDict))*fact
    getC2Func = re.compile("C2 := ([\w *+-/]+)[,}]")
    getC2=lambda aLine: float(eval(getC2Func.findall(aLine)[0], {}, decDict))*fact
    getDelayF = re.compile("Delay: ([\w+-/*]+)\n")
    getDelay=lambda aLine: float(eval(getDelayF.findall(aLine)[0], {}, decDict))*fact
    C1SetFunc = re.compile(".*C1 :=.*")
    C1Set = lambda aLine: C1SetFunc.search(aLine)!=None
    C2SetFunc = re.compile(".*C2 :=.*")
    C2Set = lambda aLine: C2SetFunc.search(aLine)!=None 
    
    
    for akey in tDict.keys():
        tDict[akey].tW = [ ]
        tDict[akey].pState = initState
        tDict[akey].clocks = np.zeros((2,))
        tDict[akey].getcState = lambda  pState, aLine: re.findall("{0}.{1}->{0}.(.*?) ".format(tDict[akey].name, pState), aLine)[0]
        tDict[akey].deltaT = 0.0
        
    
    #Initialize
    aLine = aFile.readline()
    
#     while not len(re.findall('State', aLine)):
#         aLine = aFile.readline()
#     #Skip the first state since its only used for synchronization
    #Read the first transition leading all timed automats to real states
    while len(aLine):
        #print(aLine)
        #while not len(re.findall('Transitions', aLine)) and len(aLine):
        #    aLine = aFile.readline()
        if len(re.findall('Delay', aLine)):
            thisDelay = getDelay(aLine)
            #Let the time pass
            for akey in tDict.keys():
                tDict[akey].clocks = tDict[akey].clocks + thisDelay
                tDict[akey].deltaT = tDict[akey].deltaT + thisDelay
                    
        #Found a transition->Search for the values till next state
        if len(re.findall('Transitions', aLine)) and len(aLine):
            while not aLine=='\n' and len(aLine):
                aLine = aFile.readline()
                #print(aLine)
                for akey in tDict.keys():
                    try:                    
                        newState = tDict[akey].getcState(tDict[akey].pState, aLine)
                        if C1Set(aLine):
                            newC1 = getC1(aLine)
                        else:
                            newC1 = tDict[akey].clocks[0]
                        if C2Set(aLine):
                            newC2 = getC2(aLine)
                        else:
                            newC2 = tDict[akey].clocks[1]
                        #check for _discard_ and _stat_
                        if re.match('_stat_', newState)!=None or re.match('_discard_', newState)!=None:
                            #States having no funnel representation in the system. they will be stored specifically and treated in unguarded_sim
                            if re.match('_stat_', newState)!=None:
                                tDict[akey].tW.append( [tDict[akey].deltaT, '_stat_', [newC1, newC2]]  )
                            elif re.match('_discard_', newState)!=None:
                                tDict[akey].tW.append( [tDict[akey].deltaT, '_discard_', [newC1, newC2]]  )
                            else:
                                raise ValueError('??')
                        else:
                            if stateDict is None:
                                tDict[akey].tW.append( [tDict[akey].deltaT, newState, [newC1, newC2]]  )
                            else:
                                tDict[akey].tW.append( [tDict[akey].deltaT, stateDict[akey][newState], [newC1, newC2]]  )
                        tDict[akey].clocks[0] = newC1
                        tDict[akey].clocks[1] = newC2
                        tDict[akey].deltaT = 0.0
                        tDict[akey].pState = newState
                    except:
                        #No transition for this key
                        pass
        aLine = aFile.readline()
        
    #The file is entirely scanned -> make sure that no delays without any transitions occured
    for akey in tDict.keys():
        tDict[akey].tW.append( [tDict[akey].deltaT, None, [tDict[akey].clocks[0], tDict[akey].clocks[1]]]  )
    
    return tDict

##Loops through a real trace generated by a k,i,j style automat and returns timed words that can be simulated 
def traceToTW2(tDict, aFile, xmlFile, fact, stateDict, C1name = 'C1\[ID\]', C2name='C2\[ID\]', initState='waiting', mode = 'unguarded'):
    
    #Get all the definitions in the xmlFile
    #It is assumed that the guards on the transitions are CONSTANT, e.g. they do NOT depend on the current state of the variables
    #Declarations should (must?) with a comment
    #Declarations in the xml file need to be of the form 'const int abc = 95858...528;'
    declarationFunc = re.compile("const int ([\w *+-/=\[\]\{\}]+);")
    getDeclaration = lambda aLine: declarationFunc.findall(aLine)[0];
    declarationFuncLong = re.compile("const int ([\w *+-/=\[\]\{\}]+)\n")
    getDeclarationLong = lambda aLine: declarationFuncLong.findall(aLine)[0];
    varNameFunc = re.compile("(\w+)[\[ =]")
    getVarName = lambda aLine: varNameFunc.findall(aLine)[0];
    decDict={}
    aLine = xmlFile.readline()
    while len(aLine):
        #print(aLine)
        try:
            aDec = getDeclaration(aLine)
            fail = False
        except:
            fail=True
        if fail:
            try:
                aDec = getDeclarationLong(aLine)
                aLine = xmlFile.readline()
                while re.search(';', aLine)==None:
                    aDec=aDec+aLine
                    aLine = xmlFile.readline()
                aDec=aDec + re.findall('([\w *+-/=\[\]\{\}]+);', aLine)[0]
                fail = False
            except:
                fail = True
        if fail:
            aLine = xmlFile.readline()
            continue
            
        #aDec = aDec.replace(' ','') #Apply changement
        #exec(aDec)

        aDec = aDec.split('=')
        aDec[0] = aDec[0]+'='
        assert len(aDec)==2, 'unrecognised declaration in:\n'+aLine
        #Enable list usage
        aDec[0] = getVarName(aDec[0])
        aDec[1] = aDec[1].replace('{', '[')
        aDec[1] = aDec[1].replace('}', ']')
        aDec[0] = aDec[0].replace(' ','')
        aDec[1] = aDec[1].replace(' ','')
        decDict[aDec[0]] = eval(aDec[1], {}, decDict) #Create local variable
        #Next line
        aLine = xmlFile.readline()
        
    getDelayF = re.compile("Delay: ([\w+-/*]+)\n")
    getDelay=lambda aLine: float(eval(getDelayF.findall(aLine)[0], {}, decDict))*fact
    
    XSet = lambda name, aLine: re.search(".*{0} :=.*".format(name), aLine)!=None
    XFloatGet = lambda name, aLine, aDecDict: float(eval(re.findall( "{0} := ([\w *+-/\[\]]+)".format(name)+'[,\}]', aLine )[0], {}, aDecDict))*fact
    XIntGet = lambda name, aLine, aDecDict: int(eval(re.findall( "{0} := ([\w *+-/\[\]]+)".format(name)+'[,\}]', aLine )[0], {}, aDecDict))
    XGet = lambda name, aLine, aDecDict: eval(re.findall( "{0} := ([\w *+-/\[\]]+)".format(name)+'[,\}]', aLine )[0], {}, aDecDict)
    
    for akey in tDict.keys():
        tDict[akey].tW = [ ]
        tDict[akey].pState = initState
        tDict[akey].clocks = np.zeros((2,))
        tDict[akey].getcState = lambda  pState, aLine: re.findall("{0}.{1}->{0}.(.*?) ".format(tDict[akey].name, pState), aLine)[0]
        tDict[akey].deltaT = 0.0
        tDict[akey].kij = '000'
        tDict[akey].decs = dp(decDict)
        
    def treatTransLine(aDict, aLine):
        #Search for everything that can be set and evaluate it
        thisD = dp(aDict.decs)
        if XSet('thisTrans', aLine):
            thisD['thisTrans'] = eval(XGet('thisTrans', aLine, thisD), {}, aDict.decs) #Set new transition
        #Get k,i,j
        for name in ['tempa', 'tempb', 'tempc']:
            try:
                if XSet(name, aLine):
                    thisD[name] = XIntGet(name, aLine, thisD) #Set new kij in dict
            except:
                raise ValueError('?')
        newKIJ=False
        for name in ['k', 'i', 'j']:
            try:
                if XSet(name, aLine):
                    thisD[name] = XIntGet(name, aLine, thisD) #Set new kij in dict
                    newKIJ = True
            except:
                raise ValueError('?')
                        
        if newKIJ:
            aDict.kij = '{0:d}{1:d}{2:d}'.format( thisD['k'], thisD['i'], thisD['j'] )
        aDict.decs=thisD
        return 0
    
    #Initialize
    aLine = aFile.readline()
    
#     while not len(re.findall('State', aLine)):
#         aLine = aFile.readline()
#     #Skip the first state since its only used for synchronization
    #Read the first transition leading all timed automats to real states
    while len(aLine):
        #print(aLine)
        #while not len(re.findall('Transitions', aLine)) and len(aLine):
        #    aLine = aFile.readline()
        if len(re.findall('Delay', aLine)):
            thisDelay = getDelay(aLine)
            #Let the time pass
            for akey in tDict.keys():
                tDict[akey].clocks = tDict[akey].clocks + thisDelay
                tDict[akey].deltaT = tDict[akey].deltaT + thisDelay
                    
        #Found a transition->Search for the values till next state
        if len(re.findall('Transitions', aLine)) and len(aLine):
            while not aLine=='\n' and len(aLine):
                aLine = aFile.readline()
                #print(aLine)
                for akey in tDict.keys():
                    #try:
                    
                    try:                    
                        newState = tDict[akey].getcState(tDict[akey].pState, aLine)
                    except:
                        #No transition for this key
                        continue
                    #kij is set before because using temp C1 and C2
                    print(stateDict[tDict[akey].name][tDict[akey].kij])
                    treatTransLine(tDict[akey], aLine)#Check kij
                    
                    if XSet(C1name, aLine):
                        newC1 = XFloatGet(C1name, aLine, tDict[akey].decs)
                    else:
                        newC1 = tDict[akey].clocks[0]
                    if XSet(C2name, aLine):
                        newC2 = XFloatGet(C2name, aLine, tDict[akey].decs)
                    else:
                        newC2 = tDict[akey].clocks[1]
                    
#                     #kij is set after C1 and C2
#                     print(stateDict[tDict[akey].name][tDict[akey].kij])
#                     treatTransLine(tDict[akey], aLine)#Check kij
                    
                    #Translate kij into funnels
                    tDict[akey].tW.append( [tDict[akey].deltaT, stateDict[tDict[akey].name][tDict[akey].kij], [newC1, newC2]]  )
                    print(aLine[0:-1])
                    print([tDict[akey].deltaT, stateDict[tDict[akey].name][tDict[akey].kij], [newC1, newC2]])
                    tDict[akey].clocks[0] = newC1
                    tDict[akey].clocks[1] = newC2
                    tDict[akey].deltaT = 0.0
                    tDict[akey].pState = newState
                    #except:
                    #    raise ValueError('?')

        aLine = aFile.readline()
        
    #The file is entirely scanned -> make sure that no delays without any transitions occured
    for akey in tDict.keys():
        tDict[akey].tW.append( [tDict[akey].deltaT, None, [tDict[akey].clocks[0], tDict[akey].clocks[1]]]  )
    
    return tDict

##Merge two timed words that are separate in uppaal but one timedAutomat in our code
def mergeTW(*tW):
    
    tW = list(dp(tW))
    
    mergedWord = []
    
    def removeEmpty():
        while True:
            try:
                tW.remove([])
            except:
                break
        return 0
    
    removeEmpty()#Strip empty lists
    while len(tW):
        #Get next transition
        tNext = np.Inf
        kNext = None
        for k in range(len(tW)):
            if tW[k][0][0] < tNext:
                tNext = tW[k][0][0]
                kNext = k
        #remove the transition and add to merged if not discarded
        thisTrans = tW[kNext].pop(0)
        if ( (thisTrans[1] is None) or (thisTrans[1]!='_discard_') ) or True:
            mergedWord.append( thisTrans )
            for k in range(len(tW)):
                if k==kNext:
                    #Skip
                    continue
                tW[k][0][0] = tW[k][0][0]-tNext#Adjust the remaining time till next transition
        else:
            #Discard
            for k in range(len(tW)):
                if k==kNext:
                    #Skip
                    continue
                #tW[k][0] = tW[k][0]+tNext#Adjust the remaining time till next transition
        removeEmpty()
    return mergedWord
            
             
                
                
                
            
            
        
        
    
    ## Translate a timedAutomat into a uppaal file
    def timedAuttoUPPAALXML(tAut, xmlFile, dX = 60, dY = 60, dXf=10, dYf=10):
        
        iStr = '    '
        indLevel=0
        c = 3*''' ' '''
        c=c.replace(' ', '')
        
        def wStr(aStr):
            xmlFile.write( indLevel*iStr + aStr )
            return 0
        def bSec(aStr):
            wStr('<' + aStr + '>')
            indLevel=indLevel+1
            return 0
        def eSec(aStr):
            wStr('''</''' + aStr + '>')
            indLevel=indLevel-1
            return 0
        
        numId = 0;
        def nID():
            idStr = 'id{0:d}'.format(numId)
            numId = numId+1
            return idStr
        myID2UID={}
        UID2myID={}
        for aFunSys in tAut.funnelSysList:
            for _unused, aFun in aFunSys.funnelDict.items():
                thisID = aFun.ID
                uID = nID()
                myID2UID[thisID]=uID
                UID2myID[uID]=thisID
            
        #write basics
        wStr( '''<?xml version="1.0" encoding="utf-8"?>''' )
        wStr( '''<!DOCTYPE nta PUBLIC '-//Uppaal Team//DTD Flat System 1.1//EN' 'http://www.it.uu.se/research/group/darts/uppaal/flat-1_2.dtd'>''' )
        bSec('nta')
        bSec('''declaration//Autogenerated''')
        wStr('clock C1, C2;')
        eSec('declaration')
        
        
        
        
         
    
    