import math
import numpy as np


def arrayTimeCompute(timePR, arrayW,max_travel_time):
    aTime = np.zeros(max_travel_time, dtype=np.float64)

    
    mask = timePR < max_travel_time
    # Use boolean indexing to filter timePR and arrayW
    valid_times = timePR[mask]
    valid_weights = arrayW[mask]
    
    np.add.at(aTime, valid_times.astype(np.int32), valid_weights)
    
    return aTime


def computeAccessibilityScore():
    def computeAcc(timePReached, data,max_travel_time):
        arrayW = data['arrayPop']
        popComul = 0
        popMean = 0
        popsTime = arrayTimeCompute(timePReached, arrayW, max_travel_time)
        
        return sum(popsTime)
    return computeAcc


ListFunctionAccessibility = {
  
    'accessibility60min': computeAccessibilityScore(),
    'accessibility30min': computeAccessibilityScore(),
    'accessibility20min': computeAccessibilityScore(),
    

}
