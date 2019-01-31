#!/usr/bin/env python
#-*- coding: utf-8 -*-
'''
Created on 2019年1月29日

@author: davewli
'''

from __future__ import print_function
import numpy as np
import sklearn.preprocessing as preprocessing
import sklearn.ensemble as ensemble
import sklearn.metrics as metrics
import sklearn.model_selection as model_selection

def loadData(path):
    """
        load origin data
        return origin_data
    """
    origin_data = np.genfromtxt(path, delimiter=', ', dtype=str)
    return origin_data


def standProcessingData(train_data, test_data):
    """
       processing data, encoding label, one-hot feature, normalized
       return processing_data
    """
    train_data_size = len(train_data)
    data = np.row_stack([train_data, test_data])
    # label encoding, index -> 14
    labels = data[:, 14]
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    #print(le.classes_)
    labels = le.transform(labels).astype(int)
    
    # normalized, indexs -> [2, 10, 11, 12]
    min_max_scaler = preprocessing.MinMaxScaler()
    min_max_ixs = [2, 10, 11, 12]
    min_max_data = min_max_scaler.fit_transform(data[:, min_max_ixs])
    for i, d_ix in enumerate(min_max_ixs):
        data[:, d_ix] = min_max_data[:, i]
    
    # one hot, indexs -> [0, 1, 3, 4, 5, 6, 7, 8, 9, 13]
    categorical_features = [0, 1, 3, 4, 5, 6, 7, 8, 9, 13]
    features = data[:, :-1]
    categorical_names = {}
    # label encoding
    for feature in categorical_features:
        le = preprocessing.LabelEncoder()
        le.fit(features[:, feature])
        features[:, feature] = le.transform(features[:, feature])
        categorical_names[feature] = le.classes_
    
    train_fs = features[:train_data_size, :]
    train_lb = labels[:train_data_size]
    test_fs = features[train_data_size:, :]
    test_lb = labels[train_data_size:]
      
    posi_indexs, nega_indexs = getPosiNegaSample(train_lb)
    # split posi sample, split nega sample
    posi_features = train_fs[posi_indexs, :]
    posi_labels = train_lb[posi_indexs]
    nega_features = train_fs[nega_indexs, :]
    nega_labels = train_lb[nega_indexs]
    
    # one hot train
    encoder = preprocessing.OneHotEncoder(categorical_features=categorical_features)
    encoder.fit(features)
    test_fs = encoder.transform(test_fs).astype(float)
    return test_fs, test_lb, posi_features, posi_labels, nega_features, nega_labels, encoder
#    posi_features = encoder.transform(posi_features).astype(float)
#    nega_features = encoder.transform(nega_features).astype(float)
#    
#    print("posi_features shape -> %s" % str(posi_features.shape))
#    print("posi_labels shape -> %s" % str(posi_labels.shape))
#    print("nega_features shape -> %s" % str(nega_features.shape))
#    print("nega_labels shape -> %s" % str(nega_labels.shape))
#    return posi_features, posi_labels, nega_features, nega_labels


def nstandProcessingData(train_data, test_data):
    """
       processing data, encoding label
       return processing_data
    """
    train_data_size = len(train_data)
    data = np.row_stack([train_data, test_data])
    # label encoding, index -> 14
    labels = data[:, 14]
    le = preprocessing.LabelEncoder()
    le.fit(labels)
    #print(le.classes_)
    labels = le.transform(labels).astype(int)
    
    # cover string to num, indexs -> [0, 1, 3, 4, 5, 6, 7, 8, 9, 13]
    categorical_features = [0, 1, 3, 4, 5, 6, 7, 8, 9, 13]
    features = data[:, :-1]
    categorical_names = {}
    # label encoding
    for feature in categorical_features:
        le = preprocessing.LabelEncoder()
        le.fit(features[:, feature])
        features[:, feature] = le.transform(features[:, feature])
        categorical_names[feature] = le.classes_
    
    train_fs = features[:train_data_size, :].astype(float)
    train_lb = labels[:train_data_size]
    test_fs = features[train_data_size:, :].astype(float)
    test_lb = labels[train_data_size:]
      
    posi_indexs, nega_indexs = getPosiNegaSample(train_lb)
    # split posi sample, split nega sample
    posi_features = train_fs[posi_indexs, :]
    posi_labels = train_lb[posi_indexs]
    nega_features = train_fs[nega_indexs, :]
    nega_labels = train_lb[nega_indexs]
    
#    print(posi_features.shape)
#    print(posi_features[0])
#    print(nega_features.shape)
#    print(nega_features[0])
#    print(test_fs.shape)
#    print(test_lb[0])
    
    return test_fs, test_lb, posi_features, posi_labels, nega_features, nega_labels, None


def getData(train_path, test_path, is_nor_oh=True):
    train_data = loadData(train_path)
    test_data = loadData(test_path)
    test_fs, test_lb, posi_features, posi_labels, nega_features, nega_labels, one_hot_encoder \
    = standProcessingData(train_data, test_data) if is_nor_oh else nstandProcessingData(train_data, test_data)
    return test_fs, test_lb, posi_features, posi_labels, nega_features, nega_labels, one_hot_encoder


def getPosiNegaSample(all_labels):
    """
        split origin data to posi_data and nega_data
    """
    posi_indexs = []
    nega_indexs = []
    for ix, label in enumerate(all_labels):
        if 1.0 == label:
            posi_indexs.append(ix)
        else:
            nega_indexs.append(ix)
    return posi_indexs, nega_indexs


def randomSampling(sample, cnt):
    """
        use candidate set and sampling number generate sample set,
        
        return sample_data
    """
    indexs = []
    if cnt > len(sample):
        indexs.extend(range(0, len(sample)) * (cnt // len(sample)))
    indexs.extend(np.random.randint(len(sample), size = (cnt % len(sample))))
    return sample[indexs, :]


def sample(posi_features, posi_labels, nega_features, nega_labels, one_hot_encoder, train_sample_cnt, posi_rate, train_size):
    """
        get train set using posi_nega_rate
        return train_feature_set, train_target_set
    """
    posi_sample_cnt = int(train_sample_cnt * posi_rate)
    nega_sample_cnt = train_sample_cnt - posi_sample_cnt
    # random generate
    posi_sample = randomSampling(np.column_stack([posi_features, posi_labels]), posi_sample_cnt)
    nega_sample = randomSampling(np.column_stack([nega_features, nega_labels]), nega_sample_cnt)
    # merge
    sample = np.row_stack([posi_sample, nega_sample])
    # shuffle
    np.random.shuffle(sample)
    # split feature, label
    features = sample[:, :-1]
    labels = sample[:, -1]
    # split train set and test set
    train_features, test_features, train_labels, test_labels = model_selection.train_test_split(features, labels, train_size=train_size)
    # one hot
    if one_hot_encoder is not None:
        train_features = one_hot_encoder.transform(train_features).astype(float)
        test_features = one_hot_encoder.transform(test_features).astype(float)
    return train_features, train_labels, test_features, test_labels


def ks(pred_label, real_label):
    fpr, tpr, _thresholds = metrics.roc_curve(real_label, pred_label)
    return max(tpr-fpr)


def train(train_fetures, train_labels, test_features, test_labels, all_features, all_lables):
    train_labels = train_labels.astype(int)
    test_labels = test_labels.astype(int)
    all_lables = all_lables.astype(int)
    rf = ensemble.RandomForestClassifier(n_estimators=500)
    rf.fit(train_fetures, train_labels)
       
    test_pred = rf.predict(test_features)
    test_auc = metrics.roc_auc_score(test_labels, test_pred)
    print("test set auc -> %f" % test_auc)
    
    all_pred = rf.predict(all_features)
    all_auc = metrics.roc_auc_score(all_lables, all_pred)
    print("real set auc -> %f" % all_auc)
    
    predict_func = lambda x: rf.predict_proba(x).astype(float)
    test_pred_label = predict_func(test_features)[:,1]
    test_ks = ks(test_pred_label, test_labels)
    print("test set ks -> %f" % test_ks)
    
    all_pred_label = predict_func(all_features)[:,1]
    all_ks = ks(all_pred_label, all_lables)
    print("real set ks -> %f" % all_ks)
    
    return [test_auc, all_auc, test_ks, all_ks]


def run(train_path, test_path, is_nor_oh):
    np.random.seed(13)
    test_fs, test_lb, posi_features, posi_labels, nega_features, nega_labels, one_hot_encoder = getData(train_path, test_path, is_nor_oh)
    lab_smaple_rate = [
        ("posi:nega -> 1:1", 1.0/2, 2),
        ("posi:nega -> 1:5", 1.0/6, 6),
        ("posi:nega -> 1:10", 1.0/11, 11),
        ("posi:nega -> 1:50", 1.0/51, 51),
        ("posi:nega -> 1:100", 1.0/101, 101)
    ]
    train_size = 0.80
    
    lab_id = 1
    print("-----------------------fix model train sample count-----------------------")
    model_train_sample_size = [20000, 30000]
    for sample_size in model_train_sample_size:
        for (key, rate, _sample_up) in lab_smaple_rate:
            print("*****************Lab %d, %s, sample_size->%d****************" % (lab_id, key, sample_size))
            train_features, train_labels, test_features, test_labels = sample(posi_features, posi_labels, nega_features, nega_labels, one_hot_encoder, sample_size, rate, train_size)
            eval = train(train_features, train_labels, test_features, test_labels, test_fs, test_lb)
            print(eval)
            lab_id += 1
    
    print("-----------------------model train sample count up by sample rate-----------------------")
    model_train_sample_size = [1000, 3000, 5000]
    for sample_size in model_train_sample_size:
        for (key, rate, sample_up) in lab_smaple_rate:
            print("*****************Lab %d, %s, posi sample size->%d****************" % (lab_id, key, sample_size))
            train_features, train_labels, test_features, test_labels = sample(posi_features, posi_labels, nega_features, nega_labels, one_hot_encoder, sample_size*sample_up, rate, train_size)
            eval = train(train_features, train_labels, test_features, test_labels, test_fs, test_lb)
            print(eval)
            lab_id += 1
    return

if __name__ == '__main__':
    train_path = u"C:/Users/davewli/Desktop/Adult/adult.data"
    test_path = u"C:/Users/davewli/Desktop/Adult/adult.test"
    # data normalized, one hot
    # run(train_path, test_path, True)
    
    # data not normalized and one hot
    run(train_path, test_path, False)
