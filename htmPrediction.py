import dbio
import htmBunching
import htmClustering
import htmIO
import swings
from typing import List


def predictModel(linedir, date, cur_time):
    '''
    :param linedir:
    :param date:
    :param cur_time:
    :return: cluster, duration -- predicted values
    '''
    try:
        model = htmIO.loadFromFile("models/{}.model".format(linedir))
    except:
        print("Couldn't read the model for linedir {}. Predicting zero bunching".format(linedir))
        return -1, -1
    model_cluster, model_duration = model

    print("Prediction: ", linedir)
    datapoints = dbio.getLinedirData(linedir, date, date)
    markings = htmBunching.pointsMarking(datapoints)

    bbs: List[swings.BunchBlob]

    #htmClustering.clusterDatapoints(datapoints)

    print("Detecting bunching swings...")
    bbs = swings.swingsDetection(datapoints, markings, draw=False)
    #bbs = htmIO.runOrLoad("LOAD", 'bbs_all', lambda: swings.swingsDetection(datapoints, markings, draw=False))

    if len(bbs) == 0:
        return -1, -1
    else:
        for bb in bbs:
            min_time = 99999
            max_time = 0
            for dp in bb.datapoints:
                min_time = min(min_time, dp['time'])
                max_time = max(max_time, dp['time'])
            if min_time <= cur_time and max_time >= cur_time:
                print("Predicting...")
                clusters = swings.swingsPrediction(model_cluster, bb)
                duration = int(3600 * swings.swingsPrediction(model_duration, bb))
                duration_left = max(0, duration - (cur_time - min_time))
                return clusters[0], duration_left
        return -1, -1

    #swings.swingsClusterization(bbs, predictAfterDate=predictAfterDate)
    #swings.swingsVisualization(datapoints, markings, bbs, "BYBLOB", "")

