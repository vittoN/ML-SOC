import calendar
import time
import pandas as pd
import numpy as np  # Fundamental package for scientific computing with Python
from datetime import date, timedelta
from sklearn import preprocessing


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
