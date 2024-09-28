from sklearn.metrics import confusion_matrix
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
        # XGBoost non ha direttamente `min_samples_split`, ma puoi usare `min_child_weight`
        min_child_weight=100,
        max_depth=3,

        # n_estimators=100,
        # gamma=3,  # Puoi usare gamma per aggiungere regolarizzazione
        # class_weight='balanced'  # XGBoost gestisce classi sbilanciate in modo diverso
    ).fit(x_train, y_train)
    return model


def dumpModelWithTimeStamp(name, model):
    current_GMT = time.gmtime()
    time_stamp = calendar.timegm(current_GMT)
    joblib.dump(model, str(time_stamp) + ' - ' + name)


def dumpModel(name, model):
    joblib.dump(model, name)


def loadModel(name):
    return joblib.load(name)


esoticData = None
data = None
secLeaguesData = None
fullData = None
firstLeagueData = None


def getSecLeaguesData():
    global secLeaguesData
    if (secLeaguesData is None):
        secLeaguesData = pd.read_excel('../data/mergedSecleagues.xlsx')
    secLeaguesData['GoalCumulativeSum'] = secLeaguesData['HomeGoalsCumulative'] + \
        secLeaguesData['AwayGoalsCumulative']
    secLeaguesData['GoalCumulativeSumPrev'] = secLeaguesData['GoalCumulativeSum'].shift(
        1)
    secLeaguesData['FormRatio'] = secLeaguesData['HomeForm'] / \
        secLeaguesData['AwayForm']
    secLeaguesData['RecentFormRatio'] = secLeaguesData['HomeRecentHomeForm'] / \
        secLeaguesData['AwayRecentAwayForm']
    secLeaguesData['UltimoScontroDiretto'] = secLeaguesData['UltimoScontroDiretto'].replace(
        {'H': 1, 'A': 2, 'D': 0}).infer_objects(copy=False)

    secLeaguesData['isOver'] = np.where(
        secLeaguesData['MatchGoal'] > 2.5, 1, 0)
    secLeaguesData.replace([np.inf, -np.inf], np.nan, inplace=True)
    return secLeaguesData


def _processData(data, filepath):
    """
    Metodo generale per elaborare i dati.
    """
    if data is None:
        data = pd.read_excel(filepath)
    data['GoalCumulativeSum'] = data['HomeGoalsCumulative'] + data['AwayGoalsCumulative']
    data['GoalCumulativeSumPrev'] = data['GoalCumulativeSum'].shift(1)
    data['FormRatio'] = data['HomeForm'] / data['AwayForm']
    data['FormDiff'] = abs(data['HomeForm'] - data['AwayForm'])
    data['RecentFormRatio'] = data['HomeRecentHomeForm'] / data['AwayRecentAwayForm']
    data['RecentFormDiff'] = data['HomeRecentHomeForm'] - data['AwayRecentAwayForm']
    data['UltimoScontroDiretto'] = data['UltimoScontroDiretto'].replace(
        {'H': 1, 'A': 2, 'D': 0}).infer_objects(copy=False)
    data['EloDiff'] = abs(data['elo_home'] - data['elo_away'])
    data['isOver'] = np.where(data['MatchGoal'] > 2.5, 1, 0)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    return data


fullData = None


def getFullData():
    """
    Metodo specifico per elaborare i dati Full.
    """
    global fullData
    fullData = _processData(fullData, '../data/mergedFinalFull.xlsx')
    return fullData


esoticData = None


def getEsoticData():
    """
    Metodo specifico per elaborare i dati Esotic.
    """
    global esoticData
    esoticData = _processData(esoticData, '../data/mergedEsotic.xlsx')
    return esoticData


def getData():
    global data
    if (data is None):
        data = pd.read_excel('../data/mergedDataFull2.xlsx')
    data['GoalCumulativeSum'] = data['HomeGoalsCumulative'] + \
        data['AwayGoalsCumulative']
    data['GoalCumulativeSumPrev'] = data['GoalCumulativeSum'].shift(1)
    data['FormRatio'] = data['HomeForm']/data['AwayForm']
    data['RecentFormRatio'] = data['HomeRecentHomeForm'] / \
        data['AwayRecentAwayForm']
    data['UltimoScontroDiretto'] = data['UltimoScontroDiretto'].replace(
        {'H': 1, 'A': 2, 'D': 0}).infer_objects(copy=False)

    data['feat1'] = abs((data['HomeEwmaGoalsScored'] + data['HomeEwmaGoalsConceded']) /
                        (data['AwayEwmaGoalsScored'] + data['AwayEwmaGoalsConceded']))
    data['feat2'] = abs((data['HomeGoalsCumulative'] + data['AwayGoalsCumulative']) / (
        data['HomeGoalsConcededCumulative'] + data['AwayGoalsConcededCumulative']))

    data['feat2'].replace([np.inf, -np.inf], np.nan, inplace=True)
    data['feat1'].replace([np.inf, -np.inf], np.nan, inplace=True)

    data['isOver'] = np.where(data['MatchGoal'] > 2.5, 1, 0)
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    return data


def class_accuracy(y_true, y_pred):
    """
    Calcola l'accuratezza per ogni classe in un modello di classificazione binaria.

    Args:
      y_true: I veri valori delle etichette.
      y_pred: I valori delle etichette predetti.

    Returns:
      Un dizionario contenente l'accuratezza per ogni classe.
    """

    cm = confusion_matrix(y_true, y_pred)

    # Calcola l'accuratezza per la classe 0
    accuracy_class_0 = cm[0, 0] / (cm[0, 0] + cm[1, 0])

    # Calcola l'accuratezza per la classe 1
    accuracy_class_1 = cm[1, 1] / (cm[0, 1] + cm[1, 1])

    return {
        "Under": accuracy_class_0,
        "Over": accuracy_class_1
    }


train_over1Data = None
test_over1Data = None


def getTrainOver1Data():
    global train_over1Data
    if (train_over1Data is None):
        train_over1Data = pd.read_excel(
            '../Dataframe/Train Over Step1.xlsx')

    return train_over1Data


def getTestOver1Data():
    global test_over1Data
    if (test_over1Data is None):
        test_over1Data = pd.read_excel(
            '../Dataframe/Test Over Step1.xlsx')

    return test_over1Data


altriData = None


def getAltriData():
    """
    Metodo specifico per elaborare i dati Altri.
    """
    global altriData
    altriData = _processData(altriData, '../data/mergedAltriDati.xlsx')
    return altriData
