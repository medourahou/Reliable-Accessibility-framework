from .libAccessibility_version2 import arrayTimeCompute, ListFunctionAccessibility
from .libHex import area_geojson
import math
import time
import numpy
import numpy as np

import heapq
inf = 10000000

from numba import jit, int32,int64

def shortest_path_algoithm(stop_connections, start_stop, end_stop, start_time, hEnd):
    """
    
    
    Args:
    - stop_connections: Dictionary of stop connections
    - start_stop: Starting stop
    - end_stop: Destination stop
    - start_time: Initial departure time
    - hEnd: End of time window
    
    Returns:
    - Shortest path arrival time or None if no alternative path found
    """
    # Priority queue for path exploration
    pq = [(start_time, start_stop, [(start_stop, start_time)])]
    visited = {}
    
    while pq:
        current_time, current_stop, path = heapq.heappop(pq)
        
        # Improved visited check with time tracking
        if current_stop in visited and visited[current_stop] <= current_time:
            continue
        visited[current_stop] = current_time
        
        # Goal reached
        if current_stop == end_stop:
            return current_time
        
        # Explore connections
        if current_stop in stop_connections:
            for next_stop, arr_time in stop_connections[current_stop].items():
                # Stricter time window constraints
                if start_time <= current_time and current_time <= arr_time <= hEnd:
                    # Avoid revisiting with worse time
                    if (next_stop not in visited or 
                        arr_time < visited[next_stop]):
                        new_path = path + [(next_stop, arr_time)]
                        heapq.heappush(pq, (arr_time, next_stop, new_path))
    
    # No alternative path found
    return None

def find_reachable_points_with_times(point, P2SPos, P2STime, arrayCC, hStart, hEnd):
    # Initialize time array
    inf = 10000000
    timeP = numpy.full(len(P2SPos), inf, dtype=np.float64)
    timeP[point['pos']] = hStart
    
    # Precompute point-to-stop mapping
    point_to_stops = {
        p_i: set(stops[stops != -2]) 
        for p_i, stops in enumerate(P2SPos)
    }
    
    # Precompute stop times and connections with arrival times
    stop_connections = {}
    for conn in arrayCC:
        if hStart <= conn[0] <= hEnd:
            start_stop = int(conn[2])
            end_stop = int(conn[3])
            arr_time = int(conn[1])
            if start_stop not in stop_connections:
                stop_connections[start_stop] = {}
            if end_stop not in stop_connections[start_stop] or arr_time < stop_connections[start_stop][end_stop]:
                stop_connections[start_stop][end_stop] = arr_time
    
    # Find reachable points
    reachable_points = {point['pos']}
    points_to_check = {point['pos']}
    
    while points_to_check:
        new_points_to_check = set()
        
        for curr_point in points_to_check:
            # Get stops connected to this point and their times
            curr_stops = point_to_stops.get(curr_point, set())
            curr_point_time = timeP[curr_point]
            
            # Calculate times to reach stops from current point
            stop_arrival_times = {}
            for stop in curr_stops:
                stop_idx = list(point_to_stops[curr_point]).index(stop)
                stop_time = curr_point_time + P2STime[curr_point][stop_idx]
                stop_arrival_times[stop] = stop_time
                
                # Add stops reachable through connections
                if stop in stop_connections:
                    for next_stop, arr_time in stop_connections[stop].items():
                        # Try ICSA to find alternative path
                        icsa_time = shortest_path_algoithm(
                            stop_connections, 
                            stop, 
                            next_stop, 
                            stop_time, 
                            hEnd
                           
                        )
                        
                        # Use ICSA time if found and valid
                        if icsa_time is not None:
                            if stop_time <= icsa_time:
                                stop_arrival_times[next_stop] = min(
                                    stop_arrival_times.get(next_stop, float('inf')), 
                                    icsa_time
                                )
                        # Fallback to original connection
                        elif stop_time <= arr_time:
                            stop_arrival_times[next_stop] = arr_time
            
            # Find points connected to these reachable stops
            for p_i, stops in point_to_stops.items():
                if p_i not in reachable_points:
                    # Calculate minimum time to reach this point through any stop
                    min_time = inf
                    common_stops = stops.intersection(stop_arrival_times.keys())
                    
                    if common_stops:
                        for stop in common_stops:
                            stop_idx = list(point_to_stops[p_i]).index(stop)
                            if P2STime[p_i][stop_idx] != -2:
                                total_time = stop_arrival_times[stop] + P2STime[p_i][stop_idx]
                                min_time = min(min_time, total_time)
                        
                        if min_time < timeP[p_i]:
                            timeP[p_i] = min_time
                            new_points_to_check.add(p_i)
        
        reachable_points.update(new_points_to_check)
        points_to_check = new_points_to_check
    
    return timeP, sorted(list(reachable_points))

def computeAccessibilities(city, startTime, hEnd,arrayCC, arraySP, gtfsDB, computeIsochrone, first, day,max_travel_time, listAccessibility=['socialityScore1h','velocityScore1h']):
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
    total_points_length = len(total_points)
    
    for point in total_points:
        timeStart0 = time.time()

        timeP, reseable_points = find_reachable_points_with_times(
            point, P2SPos, P2STime, arrayCC, startTime, hEnd
        )
#timeS, timeP, arrayCC, P2PPos, P2PTime, P2SPos, P2STime, S2SPos, S2STime
        timePReached = timeP - startTime    
        toUpdate = {}
        timeStartStr = str(startTime)  
        
        timeListToSave = list(range(900, 3600*3+1, 900))
        data = {
            'areaHex': areaHex,
            'arrayPop': arrayPop,
            'timeListToSave': timeListToSave
        }

        # Calculate accessibility scores for each field with day-specific storage
        for field in listAccessibility:
            field_day = f"{field}_{day}"  # Include day in field name
            
            # Get existing scores for this field and day if they exist
            existing_scores = {}
            if not first and field_day in point:
                existing_scores = point[field_day]
            
            # Calculate new score
            new_score = ListFunctionAccessibility[field](
                timePReached, data,max_travel_time
            )
            
            # Update scores dictionary
            existing_scores[timeStartStr] = new_score
            toUpdate[field_day] = existing_scores

        # Handle isochrone computation if needed
        if computeIsochrone:
            geojson = {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": []}}
            for i, t in enumerate(timeP):
                listHex[i]['t'] = t - startTime
            geojson = reduceHexsInShell(
                listHex, 'vAvg', 
                shell=[-1, 0, 900, 1800, 2700, 3600, 4500, 5400, 6300, 7200, 9000]
            )
            gtfsDB['isochrones'].replace_one(
                {'_id': point['_id']},
                {
                    '_id': point['_id'],
                    'geojson': geojson,
                    'city': city,
                    'day': day  # Include day in isochrone document
                },
                upsert=True
            )

        # Update point with new scores
        gtfsDB['points'].update_one(
            {'_id': point['_id']},
            {'$set': toUpdate}
        )

        # Calculate and display progress
        totTime += time.time() - timeStart0
        avgT = float(totTime) / float(count + 1)
        h = int((tot - count) * avgT / (60 * 60))
        m = (tot - count) * avgT / 60 - h * 60

        count += 1
        
        # Show progress with day information
        for field in listAccessibility:
            field_day = f"{field}_{day}"
            if field_day in toUpdate and timeStartStr in toUpdate[field_day]:
                score = toUpdate[field_day][timeStartStr]
                print(f'point: {count}, {field_day}: {score:.1f}, time to finish: {h:.1f}h, {m:.1f}m', end="\r")
