import json
import os
import sys
import htmIO
import itertools
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from htmUtil import unique_key

def drawPTBoxQuery(query):
    query_str = json.dumps(query)
    datapoints = htmIO.readFromMongo(query)
    print('Query:', query_str)
    drawPTBox(datapoints, query_str)


def drawPTBox(datapoints, figName='', markings={}, maximized=True, saveFiguresToFile=False):
    '''
    Draws time/stops figure. All datapoints MUST be from the same date!
    :param datapoints: list
    :param figName:
    :param markings: dictionary {unique_key: cluster}        dictionary {cluster_num: [unique_keys_of_datapoints]} # when empty clusters will not be shown
    :return:
    '''
    estavl = 'est'
    #print('Total data points:', len(datapoints))
    datapoints = sorted(datapoints, key = lambda x : (x['gtfsTripID'], x['sequence']))

    lines_planned = []
    lines_actual = []

    minseq = sys.maxsize
    maxseq = 0
    mintime = sys.maxsize
    maxtime = 0
    maxload = 60 #getMaxLoad(datapoints) #60 #np.percentile([dp['load'] for dp in datapoints], 98)
    #print('Max load:', maxload)
    tripIds = set()

    clustered_dots = {}
    for i, g in itertools.groupby(datapoints, key=lambda x: x['gtfsTripID']):
        tripIds.add(i)

        lineP = []
        lineA = []
        for dp in g:
            key = unique_key(dp)
            if key in markings:
                cluster = markings[key]
                curdots = clustered_dots.get(cluster, [])
                curdots.append((dp['{}Arrival'.format(estavl)] / 3600, dp['sequence']))
                clustered_dots[cluster] = curdots

            if minseq > dp['sequence']:
                minseq = dp['sequence']
            if maxseq < dp['sequence']:
                maxseq = dp['sequence']
            mintime = min(mintime, dp['{}Arrival'.format(estavl)],dp['{}Departure'.format(estavl)],dp['gtfsArrival'])
            maxtime = max(maxtime, dp['{}Arrival'.format(estavl)],dp['{}Departure'.format(estavl)],dp['gtfsArrival'])

            #checks:
            # if dp['estArrival'] - dp['gtfsArrival'] != dp['estArrivalDelay']:
            #     print('UNEQUAL ARRIVAL:', dp)
            # if dp['estDeparture'] - dp['gtfsDeparture'] != dp['estDepartureDelay']:
            #     print('UNEQUAL DEPARTURE:', dp)

            load = min(1.0, dp['load'] / maxload)
            lineA.append((dp['{}Arrival'.format(estavl)] / 3600, dp['sequence'], load))
            lineA.append((dp['{}Departure'.format(estavl)] / 3600, dp['sequence'], load))
            lineP.append((dp['gtfsArrival'] / 3600, dp['sequence'], 0.8))
        lines_actual.append(lineA)
        lines_planned.append(lineP)

    #print('Trip IDs:', tripIds)

    #========================================
    #=====    plotting
    #========================================
    def tocmap(c):  # readjusting the colors to fit half of the scale of 'brg' colormap in mpl
        return 0.5 + (1-c)/2

    fig, axs = plt.subplots(figsize=(16.0, 10.0))
    norm = plt.Normalize(0, 1)

    def getClusterColors(cluster):
        '''
        :param cluster: either just cluster marking as integer, or (marking, blob_number)
        :return: (color, marker)
        '''
        if isinstance(cluster, int):
            cluster_colors = {-1: 'cx', 0: 'g+', 1: 'k^', 2: 'bo', 3: 'ms', 4: 'r*'} #,  3: 'cx', 4: 'm^', 5: 'y^', 6: 'r^'}  # 'c', 'm', 'y', 'k'
            #cluster_colors = {0: 'g+', 1: 'bo', 2: 'ks'}
            cc = cluster_colors[cluster]
            return cc[0], cc[1]
        elif len(cluster) == 2:
            markers = {-1: 'x', 0: '+', 1: 's', 2: 'o', 3: '^', 4: '*'}
            marker = markers[cluster[0]]
            colors = {-1:'g', 0:'k', 1:'b', 2: 'm', 3: 'y', 4: 'r', 5:'xkcd:violet', 6:'tab:orange', 7:'tab:pink', 8:'xkcd:gold', 9:'xkcd:chartreuse',
                      10:'xkcd:azure', 11:'xkcd:pastel blue', 12:'xkcd:aqua blue', 13:'xkcd:greyish purple', 14:'xkcd:bland'}
            color = colors[cluster[1] if cluster[1]<0 else cluster[1] % (len(colors)-1)]
            return color, marker

        else:
            print('ERROR! Wrong cluster assignment for htmPlotting.drawPTBox: ', cluster)
            exit(-1)

    for cluster, dots in clustered_dots.items():
        color, marker = getClusterColors(cluster)
        x, y = zip(*dots)
        axs.plot(x, y, color=color, marker=marker, linestyle='None')

    for color, cmap, width, lines in [  \
            ('b','Blues', 1, lines_planned), \
            ('r', 'brg', 2, lines_actual) \
            ]:
        for line in lines:
            x, y, colors = list(zip(*line))
            colors = [tocmap(c) for c in colors]
            # plt.plot(x, y, color=color)
            points = np.array([x, y]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            lc = LineCollection(segments, cmap=cmap, norm=norm)  # 'viridis'
            lc.set_array(np.array(colors))
            lc.set_linewidth(width)
            axs.add_collection(lc)

    def calctime(mintime, maxtime, step):
        '''calculates good time window to show in graph'''
        mint_tmp = mintime / 3600
        maxt_tmp = maxtime / 3600
        mint = int(mint_tmp / step) * step
        maxt = int(maxt_tmp / step) * step
        if (mint + step - mint_tmp < step * 0.06):
            mint = mint + step
        if maxt_tmp - maxt > step * 0.06:
            maxt += step
        return mint, maxt, step

    # #full
    # xst, xfin, xstep = 5.5, 25, 0.5
    #yst, yfin, ystep = 1, 20, 1
    xst, xfin, xstep = calctime(mintime, maxtime, 0.5)
    yst, yfin, ystep = minseq, maxseq, 1

    plt.axis([xst, xfin, yst, yfin])
    plt.xticks(np.arange(xst, xfin, xstep))

    xticks = axs.get_xticks().tolist()
    for i in range(len(xticks)):
        xticks[i] = '{}:{}'.format(int(xticks[i] % 24), '30' if (abs(xticks[i] - int(xticks[i]))>0.4) else '00')
    axs.set_xticklabels(xticks)

    plt.yticks(np.arange(yst, yfin, ystep))

    plt.grid(axis='y', linestyle='-', linewidth=0.3)
    plt.xlabel('Time')
    plt.ylabel('Stops sequence')
    plt.title(figName)
    if saveFiguresToFile:
        filename = "./resources/figures/{}.png".format(figName)
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename))
        plt.savefig(filename)
        plt.close()
    else:
        fig = plt.gcf()
        fig.canvas.set_window_title(figName)
        if maximized:
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        plt.show(block=True)

def markingsPlotting(datapoints, markings, only11to16=False):
    for i in range(31):
        date = '201503{}'.format(str(i+1).zfill(2))
        dps = [dp for dp in datapoints if dp['date'] == date]
        if only11to16:  # i % 2 != 0:
            dps = [dp for dp in dps if 11 * 3600 < dp['time'] < 16 * 3600]
        if len(dps) > 0:
            drawPTBox(dps, figName=date, markings=markings)

def plotHistogram(values, bins = []):
    if bins == []:
        plt.hist(values)
    else:
        plt.hist(values, bins)
    plt.show(block=True)

def plotScatter(X, Y, xx=[], yy=[]):
    axisMax = 4  # max(max(X), max(Y))
    plt.axis([0, axisMax, 0, axisMax])
    plt.scatter(X,Y)

    if len(xx) > 0:
        plt.plot(xx, yy, c='r')

    plt.show(block=True)
