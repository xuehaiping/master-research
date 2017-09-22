import math

##city parameters
radius = 0.87544195
##city list
Cities = [
#Los Angeles
["LAX",[-118.248191,34.045006]],
#New York  -73.061719, 40.821452
["NYC",[-74.003105, 40.716205]],
#miami
["MIA",[-80.194734, 25.749890]],
#Orlando
["ORL",[-81.369054, 28.537684]],
#atlanta
["ATL",[-84.387956, 33.740127]],
#Charlotte
["CHA",[-80.841435, 35.229785]],
#Nashville
["NAV",[-86.766753, 36.151462]],
#Louisville
["LOU",[-85.748268, 38.249390]],
#Cincinnati
["CIN",[-84.500767, 39.099486]],
#Indianapolis
["IND",[-86.157326, 39.778972]],
#Columbus
["COL",[-82.998694, 39.961160]],
#Detroit
["DET",[-83.044275, 42.330866]],
#Chicago
["CHI",[-87.627793, 41.874563]],
#Milwaukee
["MIL",[-87.919372, 43.040745]],
#Minneapolis
["MIN",[-93.285971, 44.982799]],
#Philadelphia
["PHI",[-75.159460, 39.945103]],
#Rhode Island
["RHI",[-71.403432, 41.823369]],
#Washington
["WAS",[-76.604185, 39.282868]],
#Pittsburgh
["PIT",[-79.988188, 40.430130]],
#Kansas City
["KAN",[-94.588492, 39.089903]],
#Denver
["DEN",[-104.940650, 39.727695]],
#Dallas
["DAL",[-96.831669, 32.776106]],
#Houston
["HOU",[-95.366488, 29.761600]],
#Phoenix
["PHX",[-112.060061, 33.466748]],
#Las Vegas
["LAV",[-115.144610, 36.168834]],
#San Francisco
["SAF",[-122.419689, 37.775005]],
#Portland
["POR",[-122.676377, 45.523670]],
#Seattle
["SEA",[-122.339670, 47.573777]]
]




#compute the distance between two points
def distance(p1, old_p2):
    p2 = [old_p2[1], old_p2[0]]
    return math.sqrt(math.pow(float(p1[0])-float(p2[0]), 2) + math.pow(float(p1[1])-float(p2[1]), 2))

#check if a point is in an circle
def in_circle(center,radius, point):
    if point is not None and distance(center, point) <= radius:
        return True
    else:
        return False

#check if the point is inside one of the cities
def city_code(cities, rad, point):
    for city in cities:
        if in_circle(city[1],rad, point):
            return city[0]
    return "None"

def udf_city_code(point):
    return city_code(Cities, radius, point)
