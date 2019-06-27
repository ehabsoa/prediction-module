import copy
import math

import statistics
import numpy as np
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn import metrics, svm
from datetime import datetime
import itertools

from sklearn.cluster import KMeans, DBSCAN
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer

from htmPlotting import drawPTBox
import collections
import numpy as np
from typing import List

class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, keys):
        self.keys = keys

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        newdata_dict = []
        for dp in data_dict:
            newdp = {}
            for key in self.keys:
                if key in dp:
                    newdp[key] = dp[key]
            newdata_dict.append(newdp)
        return newdata_dict


def __getLabelsByClustering(datapoints_train, parameters, K, datapoints_predict = []):
    '''
    :param datapoints_train: datapoints for performing clustering
    :param parameters: list of features that will be used in clustering
    :param K: number of clusters
    :param datapoints_predict: datapoints that will not be used in clustering, but that will be assigned to a cluster
                            after clustering is performed
    :return: [labels, labels_predict] -- cluster labels for [datapoints_train, datapoints_predict]
    '''

    prep = Pipeline([
        ('selector', ItemSelector(keys=parameters)),
        ('dictvect', DictVectorizer(sparse=False)),
        ('stdscaler', StandardScaler()), #with_mean=False
        ('norm', Normalizer())
    ])
    # for dp in datapoints_train:
    #     ddp = [dp]
    #     XXX = prep.fit_transform(ddp)

    X = prep.fit_transform(datapoints_train)

    est = KMeans(n_clusters=K)
    # est = DBSCAN(min_samples=10000, algorithm='ball_tree')
    est.fit(X)
    labels = est.labels_

    labels_train_predict = est.predict(X)
    assert(len(labels) == len(labels_train_predict))
    nonequal = 0
    for i in range(len(labels)):
        if labels[i] != labels_train_predict[i]:
            nonequal += 1
    print("NON EQUAL: ", nonequal, " out of ", len(labels))


    if len(datapoints_predict) > 0:
        print("PREDICT IS NOT ZERO! Predicting cluster for datapoints in 'predict'...")
        X_predict = prep.fit_transform(datapoints_predict)
        labels_predict = est.predict(X_predict)
    else:
        labels_predict = []

    return labels, labels_predict

def htmCluster(datapoints, K:int, i_parameters, params_notused = [], datapoints_predict:List[dict] = []):
    '''
    :param datapoints: list of dictionaries (each dp is a dict)
        parameters: list of keys to consider from the datapoints dictionaries, e.g. ['sequence', 'daytype', 'dwell']
        params_notused: list of parameters that will be printed in results, but will not be used during clustering
        K: number of clusters
    :return: clusters, clusters_predict
        clusters = [0, 0, 1, 2, ...] #cluster number for each datapoint

    '''

    parameters = i_parameters.copy()
    for pnu in params_notused:
        if pnu in parameters:
            parameters.remove(pnu)

    #labels = htmIO.runOrLoad(cnf.runtypeClusters, 'labels-{}'.format(K), lambda: __getLabelsByClustering(datapoints, parameters, K))
    labels, labels_predict = __getLabelsByClustering(datapoints, parameters, K, datapoints_predict)

    counter = collections.Counter(labels)
    print('CLUSTER COUNTS (total:{}):'.format(len(datapoints)))
    color_nums = [0] * K
    color_num = 0
    sorted_counter_items = sorted(counter.items(), key=lambda cl_num: cl_num[1], reverse=True)
    for cluster, num in sorted_counter_items:
        color_nums[cluster] = color_num
        print("{}: {} (clustered as: {})".format(color_num, num, cluster))
        color_num += 1

    cluster_labels = [color_nums[l] for l in labels]
    cluster_labels_predict = [color_nums[l] for l in labels_predict]

    size_all = len(datapoints)
    print('CLUSTER EXPLANATION:')
    clusters = {}
    for i in range(len(datapoints)):
        curlist = clusters.get(cluster_labels[i], [])
        curlist.append(datapoints[i])
        clusters[cluster_labels[i]] = curlist
    for cluster, dps in clusters.items():
        size = len(dps)
        print('cluster {}: {} ({:7.2f}%)'.format(cluster, size, 100*size/size_all))
        param: str
        allparams = copy.deepcopy(parameters)
        allparams.extend(params_notused)
        for param in allparams:
            strout = param + ': '
            allvals = [dp[param] for dp in dps if param in dp]
            if len(allvals) > 0 and not (param.startswith("tripsnum") and len(param) > 8):
                if (len(allvals) < len(dps)):
                    strout = strout + ' [{:4.1f}% missing]'.format(100*(1 - len(allvals)/len(dps)))
                if isinstance(allvals[0], type("")):
                    counter = collections.Counter(allvals)
                    for val, num in counter.items():
                        strout = strout + "{} [{:7.2f}] ".format(val, num / size * 100)
                else:
                    mean, std = np.mean(allvals), np.std(allvals)
                    strout = strout + '{:7.2f} +- {:7.2f}'.format(mean, std)
                print(strout)

    print('Clustering done.')
    return cluster_labels, cluster_labels_predict

# def clusterPlotting(datapoints, markings, only11to16 = cnf.from11to16):
#     for i in range(31):
#         date = '201503{}'.format(str(i+1).zfill(2))
#         dps = [dp for dp in datapoints if dp['date'] == date]
#         if only11to16:  # i % 2 != 0:
#             dps = [dp for dp in dps if 11 * 3600 < dp['time'] < 16 * 3600]
#         drawPTBox(dps, figName=date, markings=markings)

def predictBB(model, X_pred):
    y_pred = model.predict(X_pred)
    return y_pred

def classifyBB (X_train, y_train, X_test, y_test):
    model = Pipeline([
        ('dictvect', DictVectorizer()),
        ('clf', RandomForestClassifier(n_estimators=100)),
    #    ('clf', SGDClassifier(loss="hinge", penalty="l2")),
        #('clf', KNeighborsClassifier(5)),
    #    ('clf', svm.SVR()),
    ])

    model.fit(X_train, y_train)
    if len(y_test) > 0:
        y_pred = model.predict(X_test)

        ###### Classification
        print(metrics.classification_report(y_pred, y_test))
        print('Accuracy:', metrics.accuracy_score(y_pred, y_test))
        print('F1:', metrics.f1_score(y_pred, y_test, average='weighted'))
        print(metrics.confusion_matrix(y_pred, y_test))

        # ####### Regression
        # print('Variance score:',metrics.explained_variance_score(y_pred, y_test))
        # print('R2 score:', metrics.r2_score(y_pred, y_test))
        # print('mean_absolute_error:', metrics.mean_absolute_error(y_pred, y_test))
        # # print('mean_squared_error:', metrics.mean_squared_error(y_pred, y_test))
        # # print('mean_squared_log_error:', metrics.mean_squared_log_error(y_pred, y_test))
        # print('median_absolute_error:', metrics.median_absolute_error(y_pred, y_test))
    else:
        y_pred = []

    return model, y_pred

def regressBB (X_train, y_train, X_test, y_test):
    print("Regression started")
    model = Pipeline([
        ('dictvect', DictVectorizer()),
    #    ('clf', SGDClassifier(loss="hinge", penalty="l2")),
        ('clf', svm.SVR()),
        #('clf', svm.SVR(kernel='poly', degree=2, max_iter=1000)),
        #('clf', MLPRegressor()),
    ])

    model.fit(X_train, y_train)
    if len(y_test) > 0:
        print("Regression ended")
        y_pred = model.predict(X_test)
        print("Prediction ended")

        # model2 = svm.SVR(kernel='poly', degree=2, max_iter=1000)
        # model2.fit(y_pred, y_test)
        # y_pred_pred = model2.predict(y_pred)
        # y_pred = y_pred_pred

        print(y_test)
        print(list(y_pred))

        # ###### Classification
        # print(metrics.classification_report(y_pred, y_test))
        # print('Accuracy:', metrics.accuracy_score(y_pred, y_test))
        # print('F1:', metrics.f1_score(y_pred, y_test, average='weighted'))
        # print(metrics.confusion_matrix(y_pred, y_test))

        # y_pred = [y * y for y in y_pred]

        ####### Regression
        print('Variance score:',metrics.explained_variance_score(y_test, y_pred))
        print('R2 score:', metrics.r2_score(y_test, y_pred))
        print('mean_absolute_error:', metrics.mean_absolute_error(y_test, y_pred))
        print('mean_squared_error:', metrics.mean_squared_error(y_test, y_pred))
        print('mean_squared_log_error:', metrics.mean_squared_log_error(y_test, y_pred))
        print('median_absolute_error:', metrics.median_absolute_error(y_test, y_pred))
    else:
        y_pred = []

    return model, y_pred
