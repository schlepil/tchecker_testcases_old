import re
from funnel_package.utils import *


def timedAuttoConnStr(aT, convertTime, inFile, outFile):
    
    dC=[0,1,0,0,0,0];
    
    toID = lambda k, ki, kj: '{0:d}{1:d}{2:d}'.format(k, ki, kj)
    
    kMax=len(aT.funnelSysList)
    iNum=[]
    jNum=[]
    isCyc=[]
    toDict={}
    fromDict={}
    maxConn=0
    for kk in range(kMax):
        aFS = aT.funnelSysList[kk]
        iNum.append( aFS.numSubFun[0] )
        jNum.append( aFS.numSubFun[1] )
        isCyc.append( aFS.funnelSys[0][0].traj.isCycFlag )
        for ki in range( iNum[-1] ):
            for kj in range( jNum[-1] ):
                toDict[(kk,ki,kj)] = aFS.funnelSys[ki][kj].ID
                fromDict[aFS.funnelSys[ki][kj].ID] = (kk,ki,kj)
                thisConn = 0
                thisDict = aFS.funnelSys[ki][kj].transDict
                for aKey in thisDict.startKeys:
                    thisConn = thisConn + len(thisDict.startDict[aKey])
                maxConn = max(maxConn, thisConn)
    
    #Initialize lists
    iMax = max(iNum)
    jMax = max(jNum)
    tMinConv = [[[ 0 for kj in range(jMax-1) ] for ki in range(iMax)] for kk in range(kMax)]
    tInvMax = [[0 for ki in range(iMax)] for kk in range(kMax)]
    numConn = [[[ 0 for kj in range(jMax) ] for ki in range(iMax)] for kk in range(kMax)]
    allConn = [[[[ dC for kn in range(maxConn)] for kj in range(jMax)] for ki in range(iMax)] for kk in range(kMax)]
    
    #Loop through again and fill everything up
    numConnMax = 0
    for kk in range(kMax):
        aFS = aT.funnelSysList[kk]
        for ki in range( iNum[kk] ):
            for kj in range( jNum[kk] ):
                
                tF = aFS.funnelSys[ki][kj]
                thisDict = tF.transDict
                #First subfunnel of this velocity -> set invariant
                if kj==0:
                    try:
                        tInvMax[kk][ki] = int(guround(tF.invariantListExpr[1].specList[2])*convertTime-0.5)
                    except:
                        raise ValueError('?')
                #Set convergence    
                if kj<jNum[kk]-1:
                    #Converge
                    tMinConv[kk][ki][kj] = int(tF.PtoCmaxConvTime*convertTime)
                #Do the transitions
                thisConn = 0
                for aTimeKey in thisDict.startKeys:
                    for aTrans in thisDict.startDict[aTimeKey]:
                        #Only take transitions involving C1_p
                        try:
                            if len(aTrans.transGuardList) == 2 and ('C1_p' in aTrans.transGuardList[0].specList  and 'C1_p' in aTrans.transGuardList[1].specList) :
                                toKIJ = fromDict[aTrans.childID]
                                thisTrans = [ int(guround(aTrans.transGuardList[0].specList[1])*convertTime), int(guround(aTrans.transGuardList[1].specList[2])*convertTime), toKIJ[0], toKIJ[1], toKIJ[2], int(guround(aTrans.transSetList[0].specList[2])*convertTime) ]
                                print(str([kk, ki, kj])+str(thisTrans) )
                                print(str(aTrans.parentID)+','+str(aTrans.childID)+','+str(aTrans.transGuardPLOT)+','+str(aTrans.transSetPLOT))
                                allConn[kk][ki][kj][thisConn] = dp(thisTrans)
                                thisConn = thisConn + 1
                            elif len(aTrans.transGuardList) == 1 and '{0}=={1}'== aTrans.transGuardList[0].specList[0] and 'C1_p' == aTrans.transGuardList[0].specList[1]:
                                toKIJ = fromDict[aTrans.childID]
                                thisTrans = [ int(guround(aTrans.transGuardList[0].specList[2])*convertTime), int(guround(aTrans.transGuardList[0].specList[2])*convertTime), toKIJ[0], toKIJ[1], toKIJ[2], int(guround(aTrans.transSetList[0].specList[2])*convertTime) ]
                                print(str([kk, ki, kj])+str(thisTrans) )
                                print(str(aTrans.parentID)+','+str(aTrans.childID)+','+str(aTrans.transGuardPLOT)+','+str(aTrans.transSetPLOT))
                                allConn[kk][ki][kj][thisConn] = dp(thisTrans)
                                thisConn = thisConn + 1
                            else:
                                for it in aTrans.transGuardList:
                                    print(it.specList)
                        except:
                            pass
                numConn[kk][ki][kj] = dp(thisConn)
                numConnMax = max(numConnMax, thisConn)
    
    switchDict = {'__CONVERTTIME__':"{0:d}".format(convertTime), '__KMAX__':dp(kMax),  '__IMAX__':dp(iMax),  '__JMAX__':dp(jMax), '__ISCYC__':dp(isCyc), '__INUM__':dp(iNum), '__JNUM__':dp(jNum), '__TMINCONV__':dp(tMinConv), '__NUMCONNMAX__':dp(numConnMax), '__NUMCONN__':dp(numConn), '__TINVMAX__':tInvMax, '__ALLCONN__':dp(allConn)}
    
    replaceDict = {'[':'{', ']':'}', 'True':'true', 'False':'false'}
    
    for akey in switchDict.keys():
        switchDict[akey] = str(switchDict[akey])
        for akey2 in replaceDict.keys():
            switchDict[akey]=switchDict[akey].replace(akey2, replaceDict[akey2])

    #Loop through file and set values
    for lineIn in inFile:
        
        for akey in switchDict.keys():
            if re.search(akey, lineIn):
                lineIn = lineIn.replace(akey, switchDict[akey])
        outFile.writelines(lineIn)
    outFile.close()
    inFile.close()
    

def timedAuttoConnStr_detTrans(aT, convertTime, inFile, outFile):
    
    dC=[0,0,0,0,0];
    
    toID = lambda k, ki, kj: '{0:d}{1:d}{2:d}'.format(k, ki, kj)
    
    kMax=len(aT.funnelSysList)
    iNum=[]
    jNum=[]
    isCyc=[]
    toDict={}
    fromDict={}
    maxConn=0
    for kk in range(kMax):
        aFS = aT.funnelSysList[kk]
        iNum.append( aFS.numSubFun[0] )
        jNum.append( aFS.numSubFun[1] )
        isCyc.append( aFS.funnelSys[0][0].traj.isCycFlag )
        for ki in range( iNum[-1] ):
            for kj in range( jNum[-1] ):
                toDict[(kk,ki,kj)] = aFS.funnelSys[ki][kj].ID
                fromDict[aFS.funnelSys[ki][kj].ID] = (kk,ki,kj)
                thisConn = 0
                thisDict = aFS.funnelSys[ki][kj].transDict
                for aKey in thisDict.startKeys:
                    thisConn = thisConn + len(thisDict.startDict[aKey])
                maxConn = max(maxConn, thisConn)
    
    #Initialize lists
    iMax = max(iNum)
    jMax = max(jNum)
    tMinConv = [[[ 0 for kj in range(jMax-1) ] for ki in range(iMax)] for kk in range(kMax)]
    tInvMax = [[0 for ki in range(iMax)] for kk in range(kMax)]
    numConn = [[[ 0 for kj in range(jMax) ] for ki in range(iMax)] for kk in range(kMax)]
    allConn = [[[[ dC for kn in range(maxConn)] for kj in range(jMax)] for ki in range(iMax)] for kk in range(kMax)]
    
    #Loop through again and fill everything up
    numConnMax = 0
    for kk in range(kMax):
        aFS = aT.funnelSysList[kk]
        for ki in range( iNum[kk] ):
            for kj in range( jNum[kk] ):
                
                tF = aFS.funnelSys[ki][kj]
                thisDict = tF.transDict
                #First subfunnel of this velocity -> set invariant
                if kj==0:
                    try:
                        tInvMax[kk][ki] = int(guround(tF.invariantListExpr[1].specList[2])*convertTime-0.5)
                    except:
                        raise ValueError('?')
                #Set convergence    
                if kj<jNum[kk]-1:
                    #Converge
                    tMinConv[kk][ki][kj] = int(tF.PtoCmaxConvTime*convertTime)
                #Do the transitions
                thisConn = 0
                for aTimeKey in thisDict.startKeys:
                    for aTrans in thisDict.startDict[aTimeKey]:
                        #Only take transitions involving C1_p
                        try:
                            if len(aTrans.transGuardList) == 2 and ('C1_p' in aTrans.transGuardList[0].specList  and 'C1_p' in aTrans.transGuardList[1].specList) :
                                if not aTrans.childID==aTrans.parentID:
                                    #Only real transitions
                                    toKIJ = fromDict[aTrans.childID]
                                    midTime = int(guround((aTrans.transGuardList[0].specList[1]+aTrans.transGuardList[1].specList[2])/2)*convertTime)
                                    thisTrans = [ midTime, toKIJ[0], toKIJ[1], toKIJ[2], int(guround(aTrans.transSetList[0].specList[2])*convertTime) ]
                                    print(str([kk, ki, kj])+str(thisTrans) )
                                    print(str(aTrans.parentID)+','+str(aTrans.childID)+','+str(aTrans.transGuardPLOT)+','+str(aTrans.transSetPLOT))
                                    allConn[kk][ki][kj][thisConn] = dp(thisTrans)
                                    thisConn = thisConn + 1
                            else:
                                for it in aTrans.transGuardList:
                                    print(it.specList)
                        except:
                            pass
                numConn[kk][ki][kj] = dp(thisConn)
                numConnMax = max(numConnMax, thisConn)
    
    switchDict = {' __KMAX__':dp(kMax),  '__IMAX__':dp(iMax),  '__JMAX__':dp(jMax), '__ISCYC__':dp(isCyc), '__INUM__':dp(iNum), '__JNUM__':dp(jNum), '__TMINCONV__':dp(tMinConv), '__NUMCONNMAX__':dp(numConnMax), '__NUMCONN__':dp(numConn), '__TINVMAX__':tInvMax, '__ALLCONN__':dp(allConn)}
    
    replaceDict = {'[':'{', ']':'}', 'True':'true', 'False':'false'}
    
    for akey in switchDict.keys():
        switchDict[akey] = str(switchDict[akey])
        for akey2 in replaceDict.keys():
            switchDict[akey]=switchDict[akey].replace(akey2, replaceDict[akey2])

    #Loop through file and set values
    for lineIn in inFile:
        
        for akey in switchDict.keys():
            if re.search(akey, lineIn):
                lineIn = lineIn.replace(akey, switchDict[akey])
        outFile.writelines(lineIn)
    outFile.close()
    inFile.close()

