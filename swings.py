import itertools
from typing import List, Any

from sklearn.externals import joblib

import htmIO
import htmML
import htmPlotting
import numpy as np

from config import cnf
from htmUtil import unique_key

# ========
# helpers
# ========

class __NextBlobIndex():
    def __init__(self):
        self.cur_index = -1

    def getNext(self):
        self.cur_index += 1
        return self.cur_index

__nbi = __NextBlobIndex()

def __renumerate_blob_indices(blobs, reverse_blobs):
    #renumerate blob indices to make coloring them with diff colors easier, unique numbers
    renumbered_reverse_blobs = {}
    for bn, dps in reverse_blobs.items():
        next_blob_index = __nbi.getNext()
        for dp in dps:
            ukey = unique_key(dp)
            assert(blobs[ukey] == bn)
            blobs[ukey] = next_blob_index
        renumbered_reverse_blobs[next_blob_index] = dps
        #next_blob_index += 1
    return blobs, renumbered_reverse_blobs

# =========

def __find_swing_blobs(datapoints, markings):
    """
    :param datapoints:
    :return: blobnumbers, blobclusters
            blobsnumbers in a form of markings: {('20150301', 2492, 3): 0, ... };
                    dictionary {unique_key(dp): blobnumber}; date-gtfsID-sequence
            "-1" blob marking means dp does not belong to a swing
            blobclusters: bn -> list(dps)
    """

    neibSEQ = 3
    neibTRIP = 1

    datamatrix = {}  # (sequence, trip) -> dp
    for dp in datapoints:
        key = (dp['sequence'], dp['gtfsTripID'])
        datamatrix[key] = dp

    blobs = {}  # ukey -> blob_number
    reverse_blobs = {}  # blob_number -> list [ukey1, ukey2, ...]

    next_blob_index = 0

    for dp in datapoints:
        ukey = unique_key(dp)
        neibs = []  # list of neighboring datapoints
        for i_seq in range(dp['sequence']-neibSEQ, dp['sequence']+neibSEQ+1):
            for tripID in [dp['preTripID'], dp['gtfsTripID'], dp['nextTripID']]:  #         !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                if not np.isnan(tripID) and (i_seq, tripID) in datamatrix:
                    neib = datamatrix[(i_seq, tripID)]
                    neibs.append(neib)

            # for j_trips in range(tripIdxs[dp['gtfsTripID']]-neibTRIP, tripIdxs[dp['gtfsTripID']]+neibTRIP+1):
            #     if 0 <= j_trips < len(trips) and (i_seq, trips[j_trips]) in datamatrix:
            #         neib = datamatrix[(i_seq, trips[j_trips])]
            #         neibs.append(neib)

        # if at least 20 percent of neighbors bunched -> assign to a cluster
        total_bunched = 0
        total_known = 0
        all_blob_nums = set()
        for neib in neibs:
            neib_ukey = unique_key(neib)
            if markings[neib_ukey] >= 0:
                total_known += 1
                if markings[neib_ukey] > 0:  # if bunched!
                    total_bunched += 1
            if neib_ukey in blobs:
                all_blob_nums.add(blobs[neib_ukey])
        if total_bunched > 0 and total_bunched * 5 >= total_known:  # at least 20%
            # find biggest blob index, use new index if no existing indices, save other blob numbers to move to the biggest one
            proper_blob_index = -1
            if len(all_blob_nums) == 0:
                proper_blob_index = next_blob_index
                next_blob_index += 1
            else:
                maxsize = 0
                for bn in all_blob_nums:
                    if maxsize < len(reverse_blobs[bn]):
                        maxsize = len(reverse_blobs[bn])
                        proper_blob_index = bn
            assert(proper_blob_index >= 0)
            cur_blob_list = reverse_blobs.get(proper_blob_index, [])
            cur_blob_list.append(dp)
            blobs[ukey] = proper_blob_index  # maintain blobs
            for old_bn in all_blob_nums:  # maintain reverse_blobs
                if old_bn != proper_blob_index:
                    old_blob_list = reverse_blobs[old_bn]
                    for old_dp in old_blob_list:
                        old_key = unique_key(old_dp)
                        blobs[old_key] = proper_blob_index  # maintain blobs
                    cur_blob_list.extend(old_blob_list)
                    reverse_blobs.pop(old_bn)
            reverse_blobs[proper_blob_index] = cur_blob_list

    blobs, reverse_blobs = __renumerate_blob_indices(blobs, reverse_blobs)
    return blobs, reverse_blobs

def __create_markings_with_blobs (dps, markings, blobnumbers):
    """
    :param dps:
    :param markings: orinigal markings, only clusters {ukey -> clnum}
    :param blobnumbers: {ukey -> blobnumber}, unique blobnumber per blob
    :return: newmarkings, cluster and blobnumber {}
    """
    newmarkings = {}
    for dp in dps:
        key = unique_key(dp)
        blob = blobnumbers.get(key, -1)  # dp['sequence'] // 10 - 1
        newmark = (markings[key], blob)
        newmarkings[key] = newmark
    return newmarkings

def __extractTripsInOrder(initial_cluster_dps):
    counts = {}
    tuples = set()
    allTrips = set()
    nonFirstTrips = set()
    nonLastTrips = set()
    for dp in initial_cluster_dps:
        a = -1 if np.isnan(dp['preTripID']) else dp['preTripID']
        b = dp['gtfsTripID']
        c = -2 if np.isnan(dp['nextTripID']) else dp['nextTripID']
        tuples.add((a,b))
        tuples.add((b,c))
        counts[(a,b)] = counts.get((a,b), 0) + 1
        counts[(b,c)] = counts.get((b,c), 0) + 1
        allTrips.update((a,b,c))
        nonFirstTrips.update((b,c))
        nonLastTrips.update((a,b))

    # sanity check: remove circular references (happens due to data quality problems, only linedir 6-0, stops 7254,7275)
    for tp in tuples.copy():
        a,b = tp
        if (b,a) in tuples:
            if counts[(a,b)] > counts[(b,a)]:
                toremove = (b,a)
            else:
                toremove = (a,b)
            tuples.remove(toremove)
            if toremove in nonFirstTrips:
                nonFirstTrips.remove(toremove)
            if toremove in nonLastTrips:
                nonLastTrips.remove(toremove)

    trips = []
    firsts = set()
    while len(firsts) > 0 or len(tuples) > 0:
        seconds = set()
        for t in tuples:
            firsts.add(t[0])
            seconds.add(t[1])
        beginnings = firsts.difference(seconds)
        srt = sorted(beginnings)
        trips.extend(srt)
        remaining_tuples = set()
        firsts = set()
        for t in tuples:
            if not t[0] in beginnings:
                remaining_tuples.add(t)
            else:
                if t[1] not in trips:
                    firsts.add(t[1])
        tuples = remaining_tuples

    if len(allTrips) != len(trips):
        return -1
    else:
        # ====
        dpsByTrip = {}
        for dp in initial_cluster_dps:
            curDps = dpsByTrip.get(dp['gtfsTripID'], [])
            curDps.append(dp)
            dpsByTrip[dp['gtfsTripID']] = curDps

        assert(trips[0] not in dpsByTrip)
        assert (trips[-1] not in dpsByTrip)
        # =====

        firstTrips = allTrips.difference(nonFirstTrips)
        lastTrips = allTrips.difference(nonLastTrips)
        while trips[0] in firstTrips:
            trips = trips[1:]
        while trips[-1] in lastTrips:
            trips = trips[:-1]

        return trips


class BunchBlob:
    def __init__(self, tripIDs: list, tripTags: list, dpsByTrip: dict) -> None:  # datapoints
        """
        :param tripIDs: guaranteed to be in order
        :param tripTags: bunched, delayed or normal
        :param dpsByTrip: dict of participating datapoints per trip
        """
        #self.datapoints = datapoints
        self.tripIDs = tripIDs
        self.tripTags = tripTags
        self.dpsByTrip = dpsByTrip
        self.datapoints = [dp for dps in dpsByTrip.values() for dp in dps]
        self.linedir = self.datapoints[0]['linedir']
        self.date = self.datapoints[0]['date']

        self.__factors = None
        self.cluster = -1
        #self.calculateFactors()

    def __difflengths_absolute__(self):
        """
        :return: minlen, averlen, maxlen, averseqstart
        """
        lengths = []
        seq_starts = []
        for tid in self.tripIDs:
            dps = self.dpsByTrip[tid]
            lengths.append(len(dps))
            seq_starts.append(min(map(lambda dp: dp['sequence'], dps)))
        return min(lengths), sum(lengths) / len(lengths), max(lengths), sum(seq_starts) / len(seq_starts)

    def __difflengths_relative__(self):
        """
        relative to the total number of stops on this particular line
        :return: minlen, averlen, maxlen, averseqstart
        """
        minlen, averlen, maxlen, averseqstart = self.__difflengths_absolute__()
        totalseq = cnf.sequenceMaxes[self.linedir]
        return minlen/totalseq, averlen/totalseq, maxlen/totalseq, averseqstart/totalseq

    # def __bagOfTags__(self):
    #     bot = {}
    #     for x in range(len(self.tripTags)-2):
    #         tag = cnf.tagToText[self.tripTags[x]] + cnf.tagToText[self.tripTags[x+1]] + cnf.tagToText[self.tripTags[x+2]]
    #         bot[tag] = bot.get(tag, 0) + 1
    #     return bot

    def __magnitude_load__(self):
        """
        :return: maxAverLoad, averMaxLoad, maxDiffInLoads, averDiffInLoads
        """
        averLoadByTrip = []
        maxLoadByTrip = []
        minLoadByTrip = []
        for tid in self.tripIDs:
            loads = [dp['load'] for dp in self.dpsByTrip[tid]]
            averLoadByTrip.append(sum(loads)/len(loads))
            maxLoadByTrip.append(max(loads))
            minLoadByTrip.append(min(loads))

        maxAverLoad = max(averLoadByTrip)
        averMaxLoad = sum(maxLoadByTrip)/len(maxLoadByTrip)

        diffInNeibLoads = []
        for i in range(len(averLoadByTrip)-1):
            curloads = averLoadByTrip[i:i+2]
            bigger, smaller = (curloads[0], curloads[1]) if curloads[0] > curloads[1] else (curloads[1], curloads[0])
            if smaller == 0:
                if bigger == 0:
                    diffInNeibLoads.append(1)  # equal values of load
                else:
                    diffInNeibLoads.append(1/bigger)
                    # assume 1 passenger, and see the difference from 1 to 'bigger' - closer to reality than 'inf'
            else:
                diffInNeibLoads.append(smaller/bigger)

        maxDiffInLoads = max(diffInNeibLoads)
        averDiffInLoads = sum(diffInNeibLoads)/len(diffInNeibLoads)
        return maxAverLoad, averMaxLoad, maxDiffInLoads, averDiffInLoads

    def __magnitude_bunching__(self):
        """
        :return: maxDiffBunching, averDiffBunching
        """
        averDiffHws = []  # once per trip: average diff in bunching on this trip
        allDiffHws = []  # once per datapoint: diff in bunching at this point
        for tid in self.tripIDs:
            diffHws = []
            for dp in self.dpsByTrip[tid]:
                bigger, smaller = (dp['preAvlHw'], dp['nextAvlHw']) if dp['preAvlHw']>dp['nextAvlHw'] else (dp['nextAvlHw'], dp['preAvlHw'])
                curdiff = smaller/bigger
                diffHws.append(curdiff)
                allDiffHws.append(curdiff)
            averDiffHws.append(sum(diffHws)/len(diffHws))
        return max(averDiffHws), sum(allDiffHws)/len(allDiffHws)

    def calculateFactors(self):

        FACTORS_TYPE = "all"  # "all"  # "line1"

        factors = {}

        if FACTORS_TYPE == "all":
            factors['tripsnum'] = len(self.tripTags)
            times = list(map(lambda dp: dp['time'], self.datapoints)) #[dp['time'] for dp in self.datapoints if not np.isnan(dp['time'])]
            factors['duration'] = (max(times) - min(times)) / 3600
            factors['minstops'], factors['averstops'], factors['maxstops'], factors['averseqstart'] = self.__difflengths_relative__()
            factors['daytype'] = self.datapoints[0]['daytype']
            factors['timestart'] = min([dp['time'] / 3600 for dp in self.datapoints])
            factors['aver_load'] = sum(map(lambda dp: dp['load'], self.datapoints)) / len(self.datapoints)

            # magnitude of differences in passenger load
            factors['maxAverLoad'], factors['averMaxLoad'], factors['maxDiffInLoads'], factors['averDiffInLoads'] = self.__magnitude_load__()

            # magnitude of bunching
            factors['maxDiffBunching'], factors['averDiffBunching'] = self.__magnitude_bunching__()

        elif FACTORS_TYPE == "line1":
            factors['tripsnum'] = len(self.tripTags)
            times = list(map(lambda dp: dp['time'], self.datapoints)) #[dp['time'] for dp in self.datapoints if not np.isnan(dp['time'])]
            factors['duration'] = (max(times) - min(times)) / 3600
            # for i in range(30):
            #     factors['tripsnum{}'.format(i)] = len(self.tripTags)
            factors['minstops'], factors['averstops'], factors['maxstops'], factors['averseqstart'] = self.__difflengths_absolute__()
            #factors.update(self.__bagOfTags__())

            # timeofday, daytype, until the end stop, average passenger load,
            factors['daytype'] = self.datapoints[0]['daytype']
            factors['timestart'] = min([dp['time'] / 3600 for dp in self.datapoints])
            factors['untilend'] = "Yes" if len(list(filter(lambda dp : dp['sequence'] >= 39, self.datapoints))) > 0 else "No"    #  !!!!!! ONLY WORKS FOR linedir 1-1 !!!!!!
            factors['aver_load'] = sum(map(lambda dp: dp['load'], self.datapoints)) / len(self.datapoints)
        else:
            print("UNKNOWN FACTORS TYPE: ", FACTORS_TYPE)

        self.__factors = factors

    def factors(self):
        if self.__factors is None:
            self.calculateFactors()
        return self.__factors

    def setBlobCluster(self, cluster):
        self.cluster = cluster


def __analyse_one_initial_cluster(initial_cluster_dps, markings):
    '''
    initially extracted trips with bunching/delayed clusters are checked if they need to be split on several different clusters
    :param initial_cluster_dps:
    :param markings:
    :return:
    '''
    if len(initial_cluster_dps) < 7:
        return []

    cDel = cnf.cDel
    cBun = cnf.cBun
    cNorm = cnf.cNorm
    cUnk = cnf.cUnk
    minNumDefinite = 3

    tripIDs = __extractTripsInOrder(initial_cluster_dps)
    dpsByTrip = {}
    for dp in initial_cluster_dps:
        curDps = dpsByTrip.get(dp['gtfsTripID'], [])
        curDps.append(dp)
        dpsByTrip[dp['gtfsTripID']] = curDps

    #remove extra tripIDs if they cannot be found in dpsByTrip
    tripIDs = [tid for tid in tripIDs if tid in dpsByTrip.keys()]

    tripTags = []
    for trip in tripIDs:
        clusterCountDict = {cUnk: 0, cNorm: 0, cBun: 0, cDel: 0}
        for dp in dpsByTrip[trip]:
            key = unique_key(dp)
            cluster = markings[key]
            clusterCountDict[cluster] = clusterCountDict.get(cluster, 0) + 1
        #print(clusterCountDict)
        if clusterCountDict[cBun] >= minNumDefinite and clusterCountDict[cBun] > clusterCountDict[cDel]:
            tripTags.append(cBun)
        elif clusterCountDict[cDel] >= minNumDefinite and clusterCountDict[cDel] >= clusterCountDict[cBun]:
            tripTags.append(cDel)
        elif clusterCountDict[cNorm] >= minNumDefinite:
            tripTags.append(cNorm)
        else:
            tripTags.append(cUnk)
    #print('TAGS:',tripTags)
    #drawPTBox(initial_cluster_dps, figName="An initial blob", markings=markings)

    # 1. delete leading/trailing zeroes
    # 2. split whenever there is 0/-1, 0/-1
    # 3. don't add clusters with len(tripIDs) < 2

    curBbs = []

    def add_cluster(start, end):  # 'end' not including
        if end-start >= 2:  #don't add clusters with len(tripIDs) < 2
            curDpsByTrip = {tripId: dpsByTrip[tripId] for tripId in tripIDs[start:end]}
            bs = BunchBlob(tripIDs[start:end], tripTags[start:end], curDpsByTrip)
            curBbs.append(bs)

            #print('processed trip ids:', bs.tripTags)
            #drawPTBox(bs.datapoints, figName="A processed blob", markings=markings)

    assert(len(tripIDs) == len(tripTags))
    curStart = 0
    x = 0
    while x < len(tripIDs):
        if tripTags[x] in [cUnk, cNorm]:
            if x == curStart:
                curStart += 1  # remove leading zeroes
            elif x + 1 == len(tripTags) or tripTags[x+1] in [cUnk, cNorm]:
                # finish current cluster, don't add trailing zeroes
                add_cluster(curStart, x)
                curStart = x + 1
        x += 1
    add_cluster(curStart, x)

    return curBbs

def swingsDetection(datapoints, markings, draw=True):
    """
    :param datapoints: dp = [('linedir', '1-1'), ('sequence', 2), ('stopID', 9117), ('gtfsTripID', 2492),
                         ('daytype', 'weekend'), ('date', '20150301'), ('time', 81902), ('passChange', 2),
                         ('dwell', 23), ('delayArr', -13), ('preAvlHw', 900), ('nextAvlHw', 900), ('load', 2),
                         ('estArrival', 81902), ('estDeparture', 81925), ('gtfsArrival', 81915),
                         ('preTripID', 2527), ('nextTripID', 2493)]
    :param markings: {('20150301', 2492, 3): 0, ... }; dictionary {unique_key(dp): cluster}; date-gtfsID-sequence
    :return:
        bbs: List[BunchBlob] - list of bunching blobs
    """

    def getUniqueDays(gUD_dps):
        def groupfunc(dp):
            return dp['linedir'] + '-' + dp['date']

        groups = {}
        data = sorted(gUD_dps, key=groupfunc)
        for k, g in itertools.groupby(data, groupfunc):
            groups[k] = list(g)  # Store group iterator as a list
        return groups

    all_blob_clusters = []  #list of clusters (lists of dps): [[ukey1, ukey2,...], [ukey3, ukey4, ...], ...]
    #all_blob_numbers = {}  # {ukey -> bn}

    unique_days = getUniqueDays(datapoints)
    for key_linedir_day, dps in unique_days.items():
        print('Detecting swings for: ', key_linedir_day, ' with size ', len(dps))
        cur_blobnumbers, cur_blobclusters = __find_swing_blobs(dps, markings)
        all_blob_clusters.extend(cur_blobclusters.values())
        #all_blob_numbers.update(blobnumbers)

        if draw:
            newmarkings = __create_markings_with_blobs(dps, markings, cur_blobnumbers)
            htmPlotting.drawPTBox(dps, figName=key_linedir_day, markings=newmarkings)

    # === swingsAnalysis

    bbs: List[BunchBlob] = []
    for initial_cluster_dps in all_blob_clusters:
        cur_bbs = __analyse_one_initial_cluster(initial_cluster_dps, markings)
        bbs.extend(cur_bbs)

    print("TOTAL BUNCHBLOBS:", len(bbs))

    bbs_fixed = []
    for bb in bbs:
        factors = bb.factors()
        is_ok = True
        for factor in list(factors.values()):
            if not isinstance(factor,str) and np.isnan(factor):
                is_ok = False
        if is_ok:
            bbs_fixed.append(bb)

    print("TOTAL BUNCHBLOBS NO NANS:", len(bbs_fixed))
    return bbs_fixed

def swingsClusterization(bunchBlobs: List[BunchBlob], predictAfterDate: str = '30001231'):
    # cluster them
    bunchBlobs_before24 = [b for b in bunchBlobs if b.datapoints[0]['date'] <= predictAfterDate]
    bunchBlobs_after24 = [b for b in bunchBlobs if b.datapoints[0]['date'] > predictAfterDate]

    bbs_datapoints_predict = [b.factors() for b in bunchBlobs_after24]
    bbs_datapoints = [b.factors() for b in bunchBlobs_before24]
    parameters = set()
    cl_factors: dict  # {factor: value for each bunch blob}
    for cl_factors in bbs_datapoints:
        for param in cl_factors.keys():
            parameters.add(param)

    # parameters.remove('minstops')
    # parameters.remove('maxstops')
    print('PARAMETERS:', parameters)
    params_notused = ['minstops', 'maxstops', 'daytype', 'untilend']
    print('PARAMETERS NOT USED:', params_notused)

    parameters = list(parameters)
    clK = 4
    labels, labels_predict = htmML.htmCluster(bbs_datapoints, clK, parameters, params_notused, bbs_datapoints_predict)
    assert(len(labels) == len(bunchBlobs_before24))
    for i in range(len(labels)):
        bunchBlobs_before24[i].setBlobCluster(labels[i])
    assert (len(labels_predict) == len(bunchBlobs_after24))
    for i in range(len(labels_predict)):
        bunchBlobs_after24[i].setBlobCluster(labels_predict[i])


def swingsPrediction(model, bb:BunchBlob):
    date = bb.datapoints[0]['date']
    alltimes = list(map(lambda dp: dp['time'], bb.datapoints))
    timestart, timesend = min(alltimes), max(alltimes)

    remaining_datapoints = [dp for dp in bb.datapoints if dp['time'] < timestart + 3600]

    remTripIDsUnordered = set([dp['gtfsTripID'] for dp in remaining_datapoints])
    # remTripIDs = [tripID for tripID in bb.tripIDs if tripID in remTripIDsUnordered]
    remTripIDs, remTripTags = [], []
    for i in range(len(bb.tripIDs)):
        if bb.tripIDs[i] in remTripIDsUnordered:
            remTripIDs.append(bb.tripIDs[i])
            remTripTags.append(bb.tripTags[i])
    remDpsByTrip = {}
    for dp in remaining_datapoints:
        cur_dps = remDpsByTrip.get(dp['gtfsTripID'], [])
        cur_dps.append(dp)
        remDpsByTrip[dp['gtfsTripID']] = cur_dps

    remBlob = BunchBlob(remTripIDs, remTripTags, remDpsByTrip)
    try:
        factors = remBlob.factors()
    except:
        print("ERROR at processing remaining BunchBlob!")
        return -1

    labels_pred = htmML.predictBB(model, [factors])
    return labels_pred

def swingsTraining(datapoints:List[dict], bunchBlobs:List[BunchBlob], markings:List[dict], predictAfterDate='20150332', labelType='cluster'):
    '''
    :param datapoints:
    :param bunchBlobs:
    :param markings:
    :param predictAfterDate:
    :param labelType: 'cluster' or 'duration'
    :return:
    '''
    bbFactors_train = []
    labels_train = []
    bbFactors_test = []
    labels_test = []

    # def getSeqInfo(remaining_datapoints):
    #     byTrip = {}
    #     for dp in remaining_datapoints:
    #         trip = dp['gtfsTripID']
    #         total, minseq = byTrip.get(trip, (0,0))
    #         total += 1
    #         minseq = min(minseq, dp['sequence'])
    #         byTrip[trip] = (total, minseq)
    #     totals, minseqs = zip(*byTrip.values())
    #     return sum(totals) / len(totals), sum(minseqs) / len(minseqs)

    bb: BunchBlob
    for bb in bunchBlobs:
        date = bb.datapoints[0]['date']
        alltimes = list(map(lambda dp: dp['time'], bb.datapoints))
        timestart, timesend = min(alltimes), max(alltimes)

        remaining_datapoints = [dp for dp in bb.datapoints if dp['time'] < timestart + 3600]

        remTripIDsUnordered = set([dp['gtfsTripID'] for dp in remaining_datapoints])
        #remTripIDs = [tripID for tripID in bb.tripIDs if tripID in remTripIDsUnordered]
        remTripIDs, remTripTags = [], []
        for i in range(len(bb.tripIDs)):
            if bb.tripIDs[i] in remTripIDsUnordered:
                remTripIDs.append(bb.tripIDs[i])
                remTripTags.append(bb.tripTags[i])
        remDpsByTrip = {}
        for dp in remaining_datapoints:
            cur_dps = remDpsByTrip.get(dp['gtfsTripID'], [])
            cur_dps.append(dp)
            remDpsByTrip[dp['gtfsTripID']] = cur_dps

        remBlob = BunchBlob(remTripIDs, remTripTags, remDpsByTrip)

        try:
            factors = remBlob.factors()

            if labelType == 'cluster':
                label = bb.cluster
            elif labelType == 'duration':
                label = (timesend - timestart) / 3600
            else:
                print("ERROR! Unknown label: ", label)

            if date <= predictAfterDate:
                bbFactors_train.append(factors)
                labels_train.append(label)
            else:
                bbFactors_test.append(factors)
                labels_test.append(label)
        except:
            print("Cannot extract factors from blob (removing): ", remBlob.linedir, remBlob.date, remBlob.tripIDs)
            # htmPlotting.drawPTBox(bb.datapoints, markings=markings, maximized=False)
            # htmPlotting.drawPTBox(remBlob.datapoints, markings=markings, maximized=False)
            # factors = {}

    if labelType == 'cluster':
        model, labels_pred = htmML.classifyBB(bbFactors_train, labels_train, bbFactors_test, labels_test)
        return model
    if labelType == 'duration':
        model, labels_pred = htmML.regressBB(bbFactors_train, labels_train, bbFactors_test, labels_test)
        # diff = [labels_test[i]-labels_pred[i] for i in range(len(labels_test))]
        # #print("DIFFERENCE:", diff)
        # print('PREDICTION DIFFERENCE mean, std:', np.mean(diff), np.std(diff))
        # bin_step = 0.25  # in hours
        # bins = [x * bin_step for x in list(range(int(min(diff) // bin_step), int(max(diff) // bin_step) + 1))]
        # htmPlotting.plotHistogram(diff, bins)
        # htmPlotting.plotScatter(labels_pred, labels_test)
        return model

        # if False:  #save data to files
        #     allfeats = bbFactors_train.copy()
        #     allfeats.extend(bbFactors_test)
        #     alllabs = labels_train.copy()
        #     alllabs.extend(labels_test)
        #     alldata = allfeats, alllabs
        #     joblib.dump(alldata, 'resources/durations_for_prediction.pkl')
        #     htmIO.saveOutputToCSV("regression_pred_test", list(zip(labels_pred, labels_test)))


def swingsVisualization(datapoints, markings, bunchBlobs, vistype, dirname="blobs"):
    """
    :param vistype: "BYDAY", "BYMIDDAY", "BYBLOB"
    """
    if vistype in ["BYDAY", "BYMIDDAY"]:
        blobnumbers = {}
        bb: BunchBlob
        clusternum = 0
        for bb in bunchBlobs:
            for dp in bb.datapoints:
                if bb.cluster == -1:
                    blobnumbers[unique_key(dp)] = clusternum
                else:
                    blobnumbers[unique_key(dp)] = bb.cluster
            clusternum += 1

        newmarkings = __create_markings_with_blobs(datapoints, markings, blobnumbers)
        if vistype == "BYDAY":
            htmPlotting.markingsPlotting(datapoints, newmarkings, only11to16=False)
        else:
            htmPlotting.markingsPlotting(datapoints, newmarkings, only11to16=True)
    elif vistype == "BYBLOB":
        id = 0
        for bb in bunchBlobs:
            htmPlotting.drawPTBox(bb.datapoints, figName="{}/{}/{}".format(dirname, bb.cluster, id), markings=markings, maximized=False)
            id += 1

    else:
        print("ERROR! Unknown visualization type: ", vistype)
