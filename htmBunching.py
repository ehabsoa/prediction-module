from config import cnf
from htmUtil import unique_key



def pointsMarking(datapoints):
    '''
    :param datapoints: list
    :return: markings: dictionary {unique_key: cluster}
    '''

    HEADWAY_DIFF_FACTOR = 0.6

    markings = {}
    for dp in datapoints:

        # relative headways logic
        if dp['preAvlHw'] == 0 or dp['nextAvlHw'] == 0:
            cl = cnf.cNorm  # TODO: even better, compare avlHw with plannedHw (e.g. if pre=0, comp nextAvl<>nextPlanned)
        elif dp['preAvlHw'] * HEADWAY_DIFF_FACTOR > dp['nextAvlHw']:  # delayed
            cl = cnf.cDel
        elif dp['nextAvlHw'] * HEADWAY_DIFF_FACTOR > dp['preAvlHw']:  # bunched
            cl = cnf.cBun
        else:
            cl = cnf.cNorm
        markings[unique_key(dp)] = cl

    return markings


# def pointsMarking(datapoints):    # for the case of dictionary {cluster_num: [unique_keys_of_datapoints]}
#     '''
#     :param datapoints: list
#     :return: markings: dictionary {cluster_num: [unique_keys_of_datapoints]} # when empty clusters will not be shown
#     '''
#     markings = {0: [], 1: [], 2: []}
#     for dp in datapoints:
#
#         # relative headways logic
#         if dp['preAvlHw'] * 0.7 > dp['nextAvlHw']:  # delayed
#             cl = 1
#         elif dp['nextAvlHw'] * 0.7 > dp['preAvlHw']:  # bunched
#             cl = 2
#         else:
#             cl = 0
#         curlist = markings[cl]
#         curlist.append(unique_key(dp))
#         markings[cl] = curlist
#
#     return markings

