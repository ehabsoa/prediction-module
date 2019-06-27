import dbio
import htmBunching
import htmClustering
import htmIO
import swings
from typing import List


def trainModel(linedir, from_date='00000000', to_date='30001231'):
    print("Training: ", linedir)
    datapoints = dbio.getLinedirData(linedir, from_date, to_date)
    markings = htmBunching.pointsMarking(datapoints)

    bbs: List[swings.BunchBlob]

    #htmClustering.clusterDatapoints(datapoints)

    print("Detecting bunching swings...")
    bbs = swings.swingsDetection(datapoints, markings, draw=False)
    #bbs = htmIO.runOrLoad("LOAD", 'bbs_all', lambda: swings.swingsDetection(datapoints, markings, draw=False))

    if len(bbs) < 4:
        print("Not enough bunching occurences for linedir {}, model is not trained".format(linedir))
    else:
        print("Loaded BBS. Clustering bunching formations...")
        #predictAfterDate = '20150324'
        predictAfterDate = to_date
        swings.swingsClusterization(bbs, predictAfterDate=predictAfterDate)
        #swings.swingsVisualization(datapoints, markings, bbs, "BYBLOB", "")
        print("Predicting...")
        model_cluster = swings.swingsTraining(datapoints, bbs, markings, predictAfterDate=predictAfterDate, labelType='cluster')
        model_duration = swings.swingsTraining(datapoints, bbs, markings, predictAfterDate=predictAfterDate, labelType='duration')
        model = (model_cluster, model_duration)
        htmIO.saveToFile(model, "models/{}.model".format(linedir))

