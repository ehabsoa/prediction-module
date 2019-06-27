import scipy.io as sio
import pandas as pd
import numpy as np
import pymongo
import os
import json
import datetime
import csv
from sklearn.externals import joblib

class MongoHTM():
    def __init__(self):
        self.client = pymongo.MongoClient('mongodb://localhost:27020/')
        self.db = self.client['my-trac']
        self.htm = self.db['htm-2015-03']

db = MongoHTM()

def readAsDF(files):
    '''
    read files from disc as DataFrame
    :param files: list of *.mat files to open
    :return: a single DataFrame
    '''

    def processFile(file):
        mat = sio.loadmat(file)
        ndTripDetail = mat['TripDetail']  # mdata
        ndTripSummary = mat['TripSummary']
        dtypeTD = ndTripDetail.dtype  # mdtype
        ndata = {n: ndTripDetail[n][0] for n in dtypeTD.names}
        columns = list(ndata.keys())
        df = pd.DataFrame(ndata, columns=columns)
        return df

    frames = []
    if type(files) == type([]):
        for file in files:
            df1 = processFile(file)
            frames.append(df1)
        df = pd.concat(frames)
    else:
        df = processFile(files)

    return df

def parseDingFile(file):
    '''
    extracts additional fields from the name of files, created by Ding Luo, like line9_20150331.mat
    :param file: file name
    :return: (line, mode, date), where:
     line - number of the PT line
     mode - 'bus'/'tram'
     date - date of the data
    '''
    lineNStr, dateStr, _ = file.replace('_','.').split('.')
    line = int(lineNStr[4:])
    if line in [18, 21, 22, 23, 24, 25, 26, 28]: #Den Haag, HTM
        mode = 'bus'
    elif line in [2, 3, 4, 19, 1, 6, 9, 11, 12, 15, 16, 17]: #Den Haag, HTM
        mode = 'tram'
    else:
        mode = 'unknown'
        print('ERROR! Unknown mode of transport for ' + lineNStr)
    return line, mode, dateStr


def importToMongo(dir):
    '''
    Script to put Ding's files (like line9_20150331.mat) into MongoDB, only needs to be run once
    :param dir: directory with files
    '''
    for file in os.listdir(dir):

        (line, mode, date) = parseDingFile(file)

        df = readAsDF(dir + '/' + file)
        asdict = df.to_dict('records')
        datapoints = []
        for record in asdict:
            dp = {
                'line': line,
                'mode': mode,
                'date': date
            }
            try:
                for k,v in record.items():
                    if k in ['tapInTimes','tapOutTimes']:
                        if len(v) == 0:
                            dp[k] = list()
                        else:
                            tapTimes = [int(val) for val in v[0]]
                            dp[k] = tapTimes
                    else:
                        if np.isnan(v[0][0]):
                            dp[k] = float(v[0][0])
                        else:
                            dp[k] = int(v[0][0])
                json.dumps(dp)
                datapoints.append(dp)
            except:
                print('ERROR:' + dp)
        db.htm.insert_many(datapoints)
        print('processed: {}; datapoints: {} '.format(file, len(datapoints)))
    print('done all')

def readFromMongo(query, projection=None):
    # print(json.dumps(query))
    if projection is None:
        datapoints = list(db.htm.find(query))
    else:
        datapoints = list(db.htm.find(query, projection))
    # print('Total data points from mongo:', len(datapoints))
    return datapoints

def runOrLoad(runtype, filename, run_function):
    if not runtype in ['RUN', 'RUNSAVE', 'LOAD']:
        print('ERROR: Unknown runtype ''{}'' for the file ''{}'''.format(runtype, filename))
        exit(-1)
    fullfilename = 'resources/{}.pkl'.format(filename)
    if runtype == 'LOAD':
        if os.path.exists(fullfilename):
            retval = joblib.load(fullfilename)
        else:
            runtype = 'RUNSAVE'
    if runtype in ['RUN', 'RUNSAVE']:
        retval = run_function()
    if runtype == 'RUNSAVE':
        if not os.path.exists(os.path.dirname(fullfilename)):
            os.makedirs(os.path.dirname(fullfilename))
        joblib.dump(retval, fullfilename)
    return retval

def saveToFile(data, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    joblib.dump(data, filename)

def loadFromFile(filename):
    return joblib.load(filename)

# if __name__ == '__main__':
#     importToMongo('C:\WORK\data\dittlab_pt_data\denhaag_201503\load_profiles_ding_TRB2018')

def saveOutputToCSV(filename, data):
    '''
    :param filename: saved to resources/filename.csv
    :param data: list of rows: [(v11, v12, v13), (v21, v22, v23), ...]
    :return:
    '''
    # with open(, 'wb') as file:
    #     wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    #     wr.writerow(mylist)
    np.savetxt("resources/{}.csv".format(filename), data, fmt='%2.6f', delimiter=',', newline='\n')
    pass

def loadDataFromCSV(filename):
    import pandas as pd
    data = pd.read_csv("resources/{}.csv".format(filename),header=None)
    y_pred = data[0].tolist()
    y_test = data[1].tolist()

    return y_pred, y_test