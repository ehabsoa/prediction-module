import sys

import numpy as np

import htmML
from config import cnf

def extend_datapoints_with_relative_avl_times(datapoints):
    '''
    :param datapoints:
    :return: datapoints with new keys: delayArrRelative, nextAvlHwRelative, preAvlHwRelative
    '''
    datapoints_ext = []
    for dp in datapoints:
        try:
            plannedHw = max(dp['nextPlannedHw'], dp['prePlannedHw'])  # for cases with unknown 0 Hw
            if plannedHw > 5000:
                plannedHw = min(dp['nextPlannedHw'], dp['prePlannedHw'])  # for cases of last trams with vey high Hw
            if plannedHw > 0 or plannedHw < 10000:
                dp['delayArrRelative'] = dp['delayArr'] / plannedHw
                dp['nextAvlHwRelative'] = dp['nextAvlHw'] / plannedHw
                dp['preAvlHwRelative'] = dp['preAvlHw'] / plannedHw
                dp['loadRelative'] = max(0.0, min(dp['load'] / cnf.loadsMaxes[dp['linedir']], 1.0))
        except:
            print(dp)

        if np.isnan(dp['dwell']) or np.isnan(dp['delayArrRelative']) or np.isnan(dp['nextAvlHwRelative']) or \
                np.isnan(dp['preAvlHwRelative']) or np.isnan(dp['load']):
            # print(dp)
            pass
        else:
            datapoints_ext.append(dp)

    return datapoints_ext

def get6s(vars_all):
    vars_filtered = [v for v in vars_all if not np.isnan(v)]
    # print(np.percentile(vars_filtered, q=[0, 2, 5, 10, 25, 50, 75, 90, 95, 98, 100]))
    mean, std = np.mean(vars_filtered), np.std(vars_filtered)
    # within6s = list(filter(lambda dp: (mean - 3 * std) < dp < (mean + 3 * std), vars_filtered))
    # print(mean, std, mean - 3*std, mean + 3*std)
    # print('Above 3 sigma:', len(vars_filtered) - len(within6s))
    # plt.hist(within6s)
    # plt.show()

    return (mean - 3 * std, mean + 3 * std)


def filter_datapoints(datapoints):
    # ['dwell', 'delayArrRelative', 'load', 'preAvlHwRelative', 'nextAvlHwRelative']
    # print('passchange')
    # changeMinNorm, changeMaxNorm     = get6s([dp['passChange'] for dp in datapoints])
    dwellMinNorm, dwellMaxNorm       = get6s([dp['dwell'] for dp in datapoints])
    dwellMinNorm = max(dwellMinNorm, 0)
    print('dwell', dwellMinNorm, dwellMaxNorm)
    delayArrMinNorm, delayArrMaxNorm = get6s([dp['delayArrRelative'] for dp in datapoints])
    print('delayArrRelative',delayArrMinNorm, delayArrMaxNorm)
    preAvlHwMinNorm, preAvlHwMaxNorm = get6s([dp['preAvlHwRelative'] for dp in datapoints])
    preAvlHwMinNorm = 0
    print('preAvlHwRelative',preAvlHwMinNorm, preAvlHwMaxNorm)
    nextAvlHwMinNorm, nextAvlHwMaxNorm = get6s([dp['nextAvlHwRelative'] for dp in datapoints])
    nextAvlHwMinNorm = 0
    print('nextAvlHwRelative', nextAvlHwMinNorm, nextAvlHwMaxNorm)
    print('load')
    loadMinNorm, loadMaxNorm = get6s([dp['load'] for dp in datapoints])

    def isWithin6s(dp):
        return dwellMinNorm < dp['dwell'] < dwellMaxNorm and \
                   delayArrMinNorm < dp['delayArrRelative'] < delayArrMaxNorm and \
                   preAvlHwMinNorm < dp['preAvlHwRelative'] < preAvlHwMaxNorm and \
                   nextAvlHwMinNorm < dp['nextAvlHwRelative'] < nextAvlHwMaxNorm

    filtered = []
    remaining = []
    for dp in datapoints:
        if isWithin6s(dp):
            filtered.append(dp)
        else:
            remaining.append(dp)
    print('total: ', len(datapoints))
    print('filtered (good): ', len(filtered))
    print('the rest (bad): ', len(remaining))
    return filtered, remaining


def clusterDatapoints(datapoints):
    print("Adding relative times to datapoints...")
    print('before: ', len(datapoints))
    datapoints = extend_datapoints_with_relative_avl_times(datapoints)
    datapoints, _ = filter_datapoints(datapoints)

    print("Start clustering datapoints...")
    parameters = ['dwell', 'delayArrRelative', 'loadRelative', 'preAvlHwRelative', 'nextAvlHwRelative']  # 'sequence', 'daytype', 'passChange',

    for K in [3, 4, 5, 6, 7]:
        print("################### ", K)
        #K = 4
        cluster_labels, _ = htmML.htmCluster(datapoints, K, parameters)

    sys.exit(0)
