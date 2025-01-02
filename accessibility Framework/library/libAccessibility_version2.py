import math
import numpy as np
from numba import jit, int32, float64

totNumTime = 1*1800

@jit()
def tDistrScore(t):
    t /= 30.
    a = 0.2
    b = 0.7
    N= 2.5
    TBus = 67.
    if(t == 0): return 0.
    return N * math.exp(-((a*TBus)/t) - t/(b*TBus))
normtDistrScore = sum([tDistrScore(t_i) for t_i in range(totNumTime)])

@jit()
def tDistrScoreGall(t):
    t /= 60.
    c = 36.
    b = 9.3
    return math.exp(c/b) * (1. - math.exp(-t/c)) * math.exp( -t/b - c * math.exp(-t/c) / b ) / b
normtDistrScoreGall = sum([tDistrScoreGall(t_i) for t_i in range(totNumTime)])

@jit()
def tDistrScore1h(t):
    if(t < 3600):
        return 1./3600.
    else:
        return 0
    return 0
normtDistrScore1h = sum([tDistrScore1h(t_i) for t_i in range(totNumTime)])

@jit()
def tDistrScore30min(t):
    if(t < 1800):
        return 1./1800.
    else:
        return 0
    return 0
normtDistrScore30min = sum([tDistrScore30min(t_i) for t_i in range(totNumTime)])

@jit()
def normed_tDistrScore(t):
    global normtDistrScore
    return tDistrScore(t) / normtDistrScore

@jit()
def normed_tDistrScoreGall(t):
    global normtDistrScoreGall
    return tDistrScoreGall(t) / normtDistrScoreGall

@jit()
def normed_tDistrScore1h(t):
    global normtDistrScore1h
    return tDistrScore1h(t) / normtDistrScore1h
    

def normed_tDistrScore30min(t):
    global normtDistrScore30min
    return tDistrScore30min(t) / normtDistrScore30min


def areaTimeCompute(timePR,max_travel_time):
    aTime = np.zeros(max_travel_time, dtype=np.float64)

    
    mask = timePR < max_travel_time
    # Use boolean indexing to filter timePR and arrayW
    valid_times = timePR[mask]
     
    np.add.at(aTime, valid_times.astype(np.int32), 1)
    
    return aTime


def arrayTimeCompute(timePR, arrayW,max_travel_time):
    aTime = np.zeros(max_travel_time, dtype=np.float64)

    
    mask = timePR < max_travel_time
    # Use boolean indexing to filter timePR and arrayW
    valid_times = timePR[mask]
    valid_weights = arrayW[mask]
    
    
    
    
    np.add.at(aTime, valid_times.astype(np.int32), valid_weights)
    
    return aTime


def computeVelocityScore(distr):
    def computeVel(timePReached, data,max_travel_time):
        areaHex = data['areaHex']
        area_new = 0
        vAvg = 0
        integralWindTime = 0
        areasTime = areaTimeCompute(timePReached,max_travel_time)
        for time_i in range(len(areasTime)):
            area_new += areasTime[time_i]*areaHex
            if time_i > 0:
                vAvg += distr(time_i) * (1./time_i)*(math.sqrt(area_new/math.pi))
                integralWindTime += distr(time_i)
        vAvg /= integralWindTime
        vAvg *= 3600.
        return vAvg
    return computeVel

def computeSocialityScore():
    def computeSoc(timePReached, data,max_travel_time):
        arrayW = data['arrayPop']
        popComul = 0
        popMean = 0
        popsTime = arrayTimeCompute(timePReached, arrayW, max_travel_time)
        
        return sum(popsTime)#popMean
    return computeSoc

def timeVelocity(timePReached, data):
    timeListToSave = data["timeListToSave"]
    areaHex = data['areaHex']
    areasTime = areaTimeCompute(timePReached)
    res = {'timeList': timeListToSave, 'velocity': []}
    for time2Save in timeListToSave:
        area = sum(areasTime[0:time2Save]) * areaHex
        res["velocity"].append((3600./time2Save)*(math.sqrt(area/math.pi)))
    return res

def timeSociality(timePReached, data):
    timeListToSave = data["timeListToSave"]
    arrayW = data['arrayPop']
    popsTime = arrayTimeCompute(timePReached, arrayW)
    res = {'timeList': timeListToSave, 'sociality': []}
    for time2Save in timeListToSave:
        pop = sum(popsTime[0:time2Save])
        res["sociality"].append(pop)
    return res

ListFunctionAccessibility = {
    'velocityScore': computeVelocityScore(normed_tDistrScore),
    'socialityScore': computeSocialityScore(),
    'velocityScoreGall': computeVelocityScore(normed_tDistrScoreGall),
    'socialityScoreGall': computeSocialityScore(),
    'velocityScore1h': computeVelocityScore(normed_tDistrScore1h),
    'socialityScore1h': computeSocialityScore(),
    
    'velocityScore30min': computeVelocityScore(normed_tDistrScore30min),
    'socialityScore30min': computeSocialityScore(),
    'socialityScore20min': computeSocialityScore(),
    'socialityScore10min': computeSocialityScore(),
    
    'timeVelocity': timeVelocity,
    'timeSociality': timeSociality
}
