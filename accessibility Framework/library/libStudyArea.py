
import shapefile as shp
def extractBoundingBoxFromShp(shapeFilePath):
    #expecting epsg:4326 to be the default crs
    shape = shp.Reader(shapeFilePath)
    raw = shape.bbox
    return [raw[0],raw[1],raw[2],raw[3]]