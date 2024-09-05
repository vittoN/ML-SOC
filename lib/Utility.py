import calendar
import time
import pandas as pd
import numpy as np  # Fundamental package for scientific computing with Python
from datetime import date, timedelta
from sklearn import preprocessing
# Train the model
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_class_weight
import joblib

def exportExcelWithTimeStamp(df, prefix, postfix):
    current_GMT = time.gmtime()
    time_stamp = calendar.timegm(current_GMT)
    df.to_excel(prefix+str(time_stamp) + postfix)


def calculate_gain_full(row, model):
    if row['Cluster'] == row['predictions'+model]:
        if row['predictions'+model] == 1 and (row['pred_prob'+model] > 0.5):
            return row['B365D']-1
        elif row['predictions'+model] == 0:
            if (row['FTR'] == 'H'):
                return row['B365H']*calculateSize1(row['B365H'], row['B365A'])-2
            else:
                return row['B365A']*calculateSize2(row['B365H'], row['B365A'])-2
    else:
        if (row['predictions'+model] == 0):
            return -2
        elif (row['predictions'+model] == 1 and (row['pred_prob'+model] > 0.5)):
            return -1


def calculateSize1(quota1, quota2):
    return 2-2*quota1/(quota1+quota2)


def calculateSize2(quota1, quota2):
    return 2*quota1/(quota1+quota2)


def calculate_gain_quotaMin(row, quotaMin):
    size = 1
    result = 0
    if (row['B365H'] >= quotaMin and row['B365A'] >= quotaMin and row['predictions'] == 0):
        size = 2
    if row['Cluster'] == row['predictions']:
        if row['predictions'] == 1 and (row['pred_prob'] > 0.5):
            return row['B365D']-1
        elif row['predictions'] == 0:
            if (row['B365H'] >= quotaMin and row['FTR'] == 'H'):
                if (size == 2):
                    return row['B365H']*calculateSize1(row['B365H'], row['B365A'])-2
                return row['B365H']-size
            elif (row['B365H'] >= quotaMin and row['FTR'] != 'H'):
                result -= 1
            if (row['B365A'] >= quotaMin and row['FTR'] == 'A'):
                if (size == 2):
                    return row['B365A']*calculateSize2(row['B365H'], row['B365A'])-2
                return row['B365A']-size
            elif (row['B365A'] >= quotaMin and row['FTR'] != 'A'):
                result -= 1
    else:
        if (row['predictions'] == 0 or (row['predictions'] == 1 and (row['pred_prob'] > 0.5))):
            result = -size
    return result


def calculate_gainHAD(row, model):
    # if row['predictions'+model] != 1:
    # if row['Cluster'] == row['predictions'+model]:
    if row['predictions'+model] == 0:
        if (row['FTR'] == 'D'):
            # if(row['B365D'] >= 1.9):
            return row['B365D']-1
        else:
            return -1
    elif row['predictions'+model] == 1:
        if (row['FTR'] == 'H'):
            # if(row['B365H'] <= 1.9):
            return row['B365H']-1
        else:
            return -1
    elif row['predictions'+model] == 2:
        if (row['FTR'] == 'A'):
            # if(row['B365A'] >= 1.9):
            return row['B365A']-1
        else:
            return -1
        # else:
        #     return -1


def calculate_gain_O25(row):
    quotaMin = 1.4
    if (row['prediction'] == 1):
        if row['B365>2.5'] > quotaMin:
            if row['MatchGoal'] > 2.5:
                return row['B365>2.5']-1
            else:
                return -1
        else:
            return 0
    elif (row['prediction'] == 0):
        if row['B365<2.5'] > quotaMin:
            if row['MatchGoal'] < 2.5:
                return row['B365<2.5']-1
            else:
                return -1
        else:
            return 0
    else:
        return 0


def trainLogRegModel(x_train, y_train):
    """
    Train a logistic regression model.

    Parameters
    ----------
    x_train : array-like of shape (n_samples, n_features)
        Training data.

    y_train : array-like of shape (n_samples,)
        Target values.

    Returns
    -------
    model : LogisticRegression
        Trained model.
    """
    reg = 0.01

    # train a logistic regression model on the training set
    model = LogisticRegression(C=1/reg, multi_class='ovr',
                               solver="liblinear", random_state=42).fit(x_train, y_train)
    return model


def trainGBoostModel(x_train, y_train):
    model = XGBClassifier(
        random_state=42,
        min_child_weight=100,  # XGBoost non ha direttamente `min_samples_split`, ma puoi usare `min_child_weight`
        max_depth=3,

        # n_estimators=100,
        # gamma=3,  # Puoi usare gamma per aggiungere regolarizzazione
        # class_weight='balanced'  # XGBoost gestisce classi sbilanciate in modo diverso
        ).fit(x_train, y_train)
    return model

def dumpModel(name, model):
    current_GMT = time.gmtime()
    time_stamp = calendar.timegm(current_GMT)
    joblib.dump(model, str(time_stamp) + ' - ' + name)

def loadModel(name):
    return joblib.load(name)