import mysql.connector as mysql
from sklearn.externals import joblib

db = mysql.connect(
    host="localhost",
    user="mytrac",
    passwd="mytrac",
    database="mytrac"
)

if __name__ == '__main__':
    datapoints, markings = joblib.load("datapoints_all.pkl")

    print(len(datapoints))

    cursor = db.cursor()
    cursor.execute("CREATE DATABASE mytrac")
    cursor.execute("DROP TABLE avlstoppoint")
    cursor.execute("CREATE TABLE avlstoppoint ("
                   "linedir VARCHAR(255), "
                   "sequence INT, "
                   "stopID INT, "
                   "gtfsTripID INT, "
                   "daytype VARCHAR(255), "
                   "pdate VARCHAR(255), "
                   "ptime INT, "
                   "passChange INT, "
                   "dwell INT, "
                   "delayArr INT, "
                   "preAvlHw INT, "
                   "nextAvlHw INT, "
                   "passLoad INT, "
                   "estArrival INT, "
                   "estDeparture INT, "
                   "gtfsArrival INT, "
                   "preTripID INT, "
                   "nextTripID INT, "
                   "prePlannedHw INT, "
                   "nextPlannedHw INT"
                   ")")

    query = "INSERT INTO avlstoppoint (linedir, sequence, stopID, gtfsTripID, daytype, pdate, ptime, passChange, dwell, delayArr, preAvlHw, nextAvlHw, passLoad, estArrival, estDeparture, gtfsArrival, preTripID, nextTripID, prePlannedHw, nextPlannedHw) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
    #values = []
    for dp in datapoints:
        vals = tuple(dp.values())
        #values.append(vals)
        try:
            cursor.execute(query, vals)
        except:
            print("Error in: ", vals)
    db.commit()
    print(cursor.rowcount, "record inserted")
