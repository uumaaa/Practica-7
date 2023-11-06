def findExtremePoints(boundary):
    north_point = None
    south_point = None
    east_point = None
    west_point = None

    for point in boundary:
        y, x = point
        if north_point is None or y < north_point[0]:
            north_point = point
        if south_point is None or y > south_point[0]:
            south_point = point
        if east_point is None or x > east_point[1]:
            east_point = point
        if west_point is None or x < west_point[1]:
            west_point = point
    return north_point, south_point, east_point, west_point


