import calendar
import time
import pandas as pd
import numpy as np # Fundamental package for scientific computing with Python
from datetime import date, timedelta
from sklearn import preprocessing

def exportExcelWithTimeStamp(df, prefix, postfix):
    current_GMT = time.gmtime()
    time_stamp = calendar.timegm(current_GMT)
    df.to_excel(prefix+str(time_stamp) + postfix)