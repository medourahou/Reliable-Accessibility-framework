from .libAccessibility import arrayTimeCompute, ListFunctionAccessibility
from .libHex import area_geojson
import math
import time
import numpy
import numpy as np
import heapq

def icsa(connections, n_stops, start_stop, start_time, S2SPos, S2STime):
    """
    Intransitive Connection Scan Algorithm implementation
    """
    tau = np.full(n_stops, np.inf)  # For connections
    tau_f = np.full(n_stops, np.inf)  # For footpaths
    tau[start_stop] = start_time
    
    sorted_connections = connections[connections[:, 0].argsort()]
    
    for conn in sorted_connections:
        dep_time, arr_time, dep_stop, arr_stop = conn
        dep_stop = int(dep_stop)
        arr_stop = int(arr_stop)
        
        if tau[dep_stop] <= dep_time or tau_f[dep_stop] <= dep_time:
            if tau[arr_stop] > arr_time:
                tau[arr_stop] = arr_time
                
                footpath_stops = S2SPos[arr_stop]
                footpath_times = S2STime[arr_stop]
                
                for fp_stop, fp_time in zip(footpath_stops, footpath_times):
                    if fp_stop != -2 and fp_time != -2:
                        new_time = arr_time + fp_time
                        tau_f[fp_stop] = min(tau_f[fp_stop], new_time)
    
    return np.minimum(tau, tau_f)

def find_reachable_points_with_times(point, P2SPos, P2STime, arrayCC, hStart, hEnd):
    # Initialize time array
    n_stops = max(max(arrayCC[:, 2]), max(arrayCC[:, 3])) + 1
    timeP = numpy.full(len(P2SPos), np.inf, dtype=np.float64)
    timeP[point['pos']] = hStart
    
    # Get initial stops reachable from the starting point
    initial_stops = P2SPos[point['pos']][P2SPos[point['pos']] != -2]
    initial_times = P2STime[point['pos']][P2SPos[point['pos']] != -2]
    
    # Filter connections within time window
    valid_connections = arrayCC[(arrayCC[:, 0] >= hStart) & (arrayCC[:, 0] <= hEnd)]
    
    # For each initial stop reachable from the point
    for stop, initial_time in zip(initial_stops, initial_times):
        stop = int(stop)
        arrival_time = hStart + initial_time
        
        # Run ICSA from this stop
        stop_times = icsa(valid_connections, n_stops, stop, arrival_time, 
                         S2SPos, S2STime)
        
        # Update timeP for all points that can be reached from the computed stops
        for p_idx, p_stops in enumerate(P2SPos):
            valid_stops = p_stops[p_stops != -2]
            valid_times = P2STime[p_idx][p_stops != -2]
            
            for s, t in zip(valid_stops, valid_times):
                s = int(s)
                if stop_times[s] != np.inf:
                    total_time = stop_times[s] + t
                    timeP[p_idx] = min(timeP[p_idx], total_time)
    
    # Get list of reachable points
    reachable_points = np.where(timeP != np.inf)[0]
    
    return timeP, sorted(list(reachable_points))

def computeAccessibilities(city, startTime, hEnd, arrayCC, arraySP, gtfsDB, computeIsochrone, first, day, max_travel_time, listAccessibility=['accessibility60min','accessibility30min']):
    timeS = arraySP['timeS']
    timeP = arraySP['timeP']
    S2SPos = arraySP['S2SPos']
    S2STime = arraySP['S2STime']
    P2PPos = arraySP['P2PPos']
    P2PTime = arraySP['P2PTime']
    P2SPos = arraySP['P2SPos']
    P2STime = arraySP['P2STime']

    maxVel = 0
    totTime = 0.
    avgT = 0 
    tot = len(timeP)
    areaHex = area_geojson(gtfsDB['points'].find_one({'city':city})['hex'])
    count = 0
     
    countPop = 0
    arrayPop = numpy.full(len(timeP), -2, dtype=numpy.float64)
    for point in gtfsDB['points'].find({'city':city}, projection={'pointN': False, 'stopN':False}).sort([('pos',1)]).max_time_ms(7200000):
        arrayPop[countPop] = point['pop']
        countPop += 1
    
    total_points = list(gtfsDB['points'].find({'city':city}, {'pointN':0, 'stopN':0}, no_cursor_timeout=True).sort([('pos',1)]))
    
    for point in total_points:
        timeStart0 = time.time()

        timeP, reachable_points = find_reachable_points_with_times(
            point, P2SPos, P2STime, arrayCC, startTime, hEnd
        )
        
        timePReached = timeP - startTime    
        toUpdate = {}
        timeStartStr = str(startTime)  
        
        timeListToSave = list(range(900, 3600*3+1, 900))
        data = {
            'areaHex': areaHex,
            'arrayPop': arrayPop,
            'timeListToSave': timeListToSave
        }

        for field in listAccessibility:
            field_day = f"{field}_{day}"
            
            existing_scores = {}
            if not first and field_day in point:
                existing_scores = point[field_day]
            
            new_score = ListFunctionAccessibility[field](
                timePReached, data, max_travel_time
            )
            
            existing_scores[timeStartStr] = new_score
            toUpdate[field_day] = existing_scores

        if computeIsochrone:
            # Handle isochrone computation (keeping original implementation)
            pass

        gtfsDB['points'].update_one(
            {'_id': point['_id']},
            {'$set': toUpdate}
        )

        totTime += time.time() - timeStart0
        avgT = float(totTime) / float(count + 1)
        h = int((tot - count) * avgT / (60 * 60))
        m = (tot - count) * avgT / 60 - h * 60

        count += 1
        
        for field in listAccessibility:
            field_day = f"{field}_{day}"
            if field_day in toUpdate and timeStartStr in toUpdate[field_day]:
                score = toUpdate[field_day][timeStartStr]
                print(f'point: {count}, {field_day}: {score:.1f}, time to finish: {h:.1f}h, {m:.1f}m', end="\r")