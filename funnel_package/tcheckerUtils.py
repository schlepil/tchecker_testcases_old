import re
from typing import List

def TAToTcheckerEdges(ta:"funnel_package.funnelClasses.timedAutomata", procName:str="rob", ctrlClkStr:str="ctrl", localClkStr:str="local")->List[str]:

    from os import linesep as nl

    locationsList = []
    transitionsList = []

    # Replace dict
    Up2Tdict = {"and":"&&", "C1_p":ctrlClkStr, "C2_p":localClkStr, "C1_c":ctrlClkStr, "C2_c":localClkStr}

    def replaceNames(aStr):
        aStr = "" if aStr is None else aStr
        for aK, aV in Up2Tdict.items():
            aStr = aStr.replace(aK, aV)
        return aStr

    # Loop over all subfunnel systems
    for aSubFunnelSys in ta.funnelSysList:
        # Over all subfunnels having the same velocity
        for aVelSubFunnelSys in aSubFunnelSys.funnelSys:
            # each subfunnel of a certain size
            for aSubFunnel in aVelSubFunnelSys:
                # Add the location with its invariant
                # Create the invariant
                thisInv=replaceNames(aSubFunnel.invariantList)

                locationsList.append( f"location:{procName}:{aSubFunnel.ID}{{invariant:{thisInv}}}{nl}" )

                transitionsList.append(f"#Transitions from {aSubFunnel.ID}{nl}")

                for aStartTime, aTransList in aSubFunnel.transDict.startDict.items():
                    for aTrans in aTransList:
                        # Get the guard
                        thisGuard = replaceNames(aTrans.transGuard)
                        thisGuard = f"provided:{thisGuard}" if thisGuard else thisGuard
                        thisSet = replaceNames(aTrans.transSet) # Here are some rdundant assignments like x=x mais bon...
                        thisSet = f"do:{thisSet}" if thisSet else thisSet
                        # get the transition
                        transitionsList.append( f"edge:{procName}:{aTrans.parentID}:{aTrans.childID}:{aTrans.event}{{{thisGuard}{':' if (thisGuard and thisSet) else ''}{thisSet}}}{nl}" )

    return locationsList, transitionsList






