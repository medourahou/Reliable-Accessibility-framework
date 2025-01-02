import pandas as pd
import numpy as np
import pyproj
import geopandas as gpd
import logging

def importcells_run(path,w_crs):
    logging.info('Importing hexagonal grid created by Public-Transport-Analysis tool')
    Xpoints, Ypoints, Xhex, Yhex = importCellsCSV(path)
    df = createCellsDF(Xpoints, Ypoints, Xhex, Yhex, w_crs)
    return df

def importCellsCSV(path):
    df = pd.read_csv(path, sep=';', header=0)
    Xpoints = []
    Ypoints = []
    Xhex = []
    Yhex = []
    for i, row in df.iterrows():
        pointStr = row['point']
        i1 = (pointStr.find('[')) + 1
        i2 = (pointStr.find(']'))
        pointStr = pointStr[i1:i2]
        point = pointStr.split(', ')
        Xpoints.append(float(point[0]))
        Ypoints.append(float(point[1]))

        polygonStr = row['hex']
        i1 = (polygonStr.find('(((')) + 2
        i2 = (polygonStr.find('))'))
        polygonStr = polygonStr[i1:i2]
        polygonCoords = np.float_(polygonStr.replace('(','').replace(')','').split(', '))
        for i in range(0,len(polygonCoords),2):
            Xhex.append(polygonCoords[i])
            Yhex.append(polygonCoords[i+1])

    return Xpoints,Ypoints,Xhex,Yhex


def createCellsDF(Xpoints,Ypoints, Xhex, Yhex, w_crs):
    Points = []
    Xproj = []
    Yproj = []
    Hexs = []

    proj = pyproj.Transformer.from_crs(4326, w_crs, always_xy=True)

    for i in range(len(Xpoints)):
        x, y = proj.transform(Xpoints[i], Ypoints[i])
        Points.append('POINT (' + str(x) + ' ' + str(y) + ')')
        Xproj.append(x)
        Yproj.append(y)
    for i in range(0,len(Xhex),7):
        wkt = 'POLYGON (('
        lim = i+6
        for i in range(i, i+7, 1):
            x, y = proj.transform(Xhex[i], Yhex[i])
            wkt = wkt + str(int(round(x, 0))) + ' ' + str(int(round(y, 0)))
            if i < lim:
                wkt = wkt + ', '
        wkt = wkt + '))'
        Hexs.append(wkt)

    gs = gpd.GeoSeries.from_wkt(Hexs, crs='EPSG:'+ str(w_crs))
    CellsDF = gpd.GeoDataFrame(pd.DataFrame(), geometry=gs, crs='EPSG:'+str(w_crs))

    CellsDF['Centroids'] = Points
    CellsDF['Centroid_X'] = Xproj
    CellsDF['Centroid_Y'] = Yproj

    return CellsDF



