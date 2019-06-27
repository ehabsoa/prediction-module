import mysql.connector as mysql

db = mysql.connect(
    host="localhost",
    user="mytrac",
    passwd="mytrac",
    database="mytrac"
)

def getDistinctLinedirs():
    '''all distinct linedirs'''
    cursor = db.cursor()
    query = "SELECT DISTINCT linedir from avlstoppoint"
    cursor.execute(query)
    linedirs = cursor.fetchall()
    return linedirs


def getLinedirData(linedir, from_date, to_date):
    '''Get all points of linedir, for dates in between from_date and to_date'''

    cursor = db.cursor()
    query = "SELECT * from avlstoppoint WHERE linedir='{}' AND pdate>='{}' AND pdate<='{}'".format(linedir, from_date, to_date)
    print(query)
    cursor.execute(query)
    list_datapoints = cursor.fetchall()
    datapoints = []
    for ldp in list_datapoints:
        dp = {
            'linedir': ldp[0],
            'sequence': ldp[1],
            'stopID': ldp[2],
            'gtfsTripID': ldp[3],
            'daytype': ldp[4],
            'date': ldp[5],
            'time': ldp[6],
            'passChange': ldp[7],
            'dwell': ldp[8],
            'delayArr': ldp[9],
            'preAvlHw': ldp[10],
            'nextAvlHw': ldp[11],
            'load': ldp[12],
            'estArrival': ldp[13],
            'estDeparture': ldp[14],
            'gtfsArrival': ldp[15],
            'preTripID': ldp[16],
            'nextTripID': ldp[17],
            'prePlannedHw': ldp[18],
            'nextPlannedHw': ldp[19]
        }
        datapoints.append(dp)
    print(len(datapoints))
    return datapoints
