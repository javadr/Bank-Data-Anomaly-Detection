import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_datetime64_any_dtype as is_datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import IsolationForest

from itertools import combinations
from datetime import datetime
from dateutil import parser

from rich import print
from rich_dataframe import prettify

import warnings
warnings.filterwarnings('ignore')

import time
import re
# set seed for reproducibility
np.random.seed(14)


def timeSince(since: float) -> str:
    s = time.time() - since
    return f"{int(s/60)}m {int(s%60)}s"


# function decorator to compute the time of execution
def timeit(func):
    def _wrap(*args, **kwargs):
        start = time.time()
        try:
            if func.__doc__: print(func.__doc__)
            return func(*args, **kwargs)
        finally:
            print(f"Done in {timeSince(start)}.\n")

    return _wrap


def highVar(df: pd.DataFrame, hvPercent=.7) -> dict:
    """ 
    Return a list containing the columns with high varibility and thier correspondig uniqueness values
    """
    hv = {}  # list of columns with high variability
    for col in df.columns:
        u, c = df[col].nunique(), df[col].count()
        if u / c > hvPercent:
            hv[col] = u / c * 100
    return hv


def colIsDate(series: pd.Series, ntries: int = 10) -> bool:
    """ 
    Checks #ntries elements of a specified column and returns True if it contains datetime element.
    """
    #courtesy to https://stackoverflow.com/questions/33204500/pandas-automatically-detect-date-columns-at-run-time
    if series.dtype != 'object':
        return False
    elif is_datetime(series.dtype):
        return True

    vals = set()
    for val in series:
        vals.add(val)
        if len(vals) > ntries:
            break

    for val in list(vals):
        try:
            if isinstance(val, (int, float)): continue
            elif isinstance(val, datetime): return True
            parser.parse(val)
            return True
        except ValueError:
            pass

    return False


def findDateCol(df: pd.DataFrame) -> list:
    """
    Returns columns' list of datetime type in a dataframe
    """
    dateCols = [c for c in df.columns if is_datetime(df[c])]
    cols = df.columns.values[df.dtypes.values == 'object']
    dateCols.extend([c for c in cols if colIsDate(df[c], ntries=10)])

    return dateCols


@timeit
def task1(
        df: pd.DataFrame,
        numericals: list = ["WITHDRAWAL AMT", "DEPOSIT AMT",
                            "BALANCE AMT"],  # numerical fields
) -> pd.DataFrame:
    """Data cleaning ...
    Normalizing numeric attributes
    Filling empty attributes
    Erasing surrounding spaces and makes all strings lowre-case
    Deleting columns of high variablity"""
    df = df.copy()

    # normalizing numerical fields & filling the empty deposit/withdrawal with 0
    for col in numericals:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        # uncomment the below if you want z-score normalization instead min-max normalization
        #df[col] = (df[col]-df[col].mean()) / df[col].std()
        df[col].fillna(0, inplace=True)

    # trimming surrounding spaces of data strings and making them in lower case
    df = df.applymap(lambda x: x.lower().strip() if isinstance(x, str) else x)

    # fill NAs with most commmon value
    for c in df.columns:
        if df[c].isnull().any():
            df[c].fillna(df[c].value_counts()[:1].index.to_list()[0],
                         inplace=True)

    # removing columns with high variability
    hv = [c for c in highVar(df) if c != "BALANCE AMT"]
    df.drop(columns=hv, inplace=True)

    return df


@timeit
def task2(df: pd.DataFrame) -> pd.DataFrame:
    """Data cleaning ...
    Reformulating the columns that contatins date format"""
    df = df.copy()
    dateCandids = findDateCol(df)

    attr = [
        'Year', 'Month', 'Week', 'Day', 'Dayofweek', 'Dayofyear',
        'Is_month_end', 'Is_month_start', 'Is_quarter_end', 'Is_quarter_start',
        'Is_year_end', 'Is_year_start'
    ]
    regex = "([0-9]+[-/][0-9]+[-/][0-9]+)"

    for dc in dateCandids:
        # extracting text part of the date filed
        df[f"{dc} text"] = df[dc].apply(
            lambda x: ' '.join(re.findall(r'([a-zA-Z_\- ]+)', x)).strip()
            if isinstance(x, str) else 'exact')
        df[f"{dc} text"] = df[f"{dc} text"].astype('category').cat.as_ordered()
        df[dc] = df[dc].apply(lambda x: re.findall(regex, x)[0]
                              if isinstance(x, str) else x)
        # converting to date time type
        df[dc] = pd.to_datetime(df[dc], errors='coerce', dayfirst=True)
        # adding more attributes from data fields
        for n in attr:
            df[f"{dc} {n}"] = getattr(df[dc].dt, n.lower())
            df[f"{dc} {n}"] = df[f"{dc} {n}"].astype(
                'category').cat.as_ordered()

    return df


@timeit
def task3(
        df: pd.DataFrame,
        numericals: list = ["WITHDRAWAL AMT", "DEPOSIT AMT",
                            "BALANCE AMT"],  # numerical fields
) -> pd.DataFrame:
    """Data cleaning ...
    Extracting numeric values and thier companion text into two separate fields"""
    df = df.copy()
    regex = r"[+-]?(?:[0-9]+(?:[.][0-9]*)?|[.][0-9]+)"

    for nc in numericals:
        text = f"{nc} text"
        df[text] = df[nc].apply(lambda x: ' '.join(
            re.findall(r'([^0-9]+)', x.replace(',', ''))).strip()
                                if isinstance(x, str) else 'exact')
        df[nc] = df[nc].apply(
            lambda x: np.nan if pd.isnull(x) or x == [] else pd.
            to_numeric(re.findall(regex, x.replace(',', ''))[0]) if isinstance(
                x, str) and len(re.findall(regex, x.replace(',', ''))) else x)
        df[text] = df[text].astype('category').cat.as_ordered()
        df[nc] = df[nc].fillna(0)
    return df


@timeit
def task4(df: pd.DataFrame,
          columns: list = [
              'Account No', 'DATE', 'TRANSACTION DETAILS', 'VALUE DATE',
              'WITHDRAWAL AMT', 'DEPOSIT AMT', 'BALANCE AMT'
          ],
          dupPercent: float = .7) -> pd.DataFrame:
    """Data cleaning ...
    Dropping duplicated transactions, if 70% of their attributes are equal"""
    ncols = len(columns)  # number of columns
    colDic = {i: col
              for i, col in enumerate(columns)
              }  # converting index to column name
    dp = int(
        ncols * dupPercent
    ) + 1  # how percentage of columns should be processed for finding duplication
    colList = ([colDic[i] for i in c] for c in combinations(range(ncols), dp)
               )  # combination of ncols attributes for duplication

    df = df.copy()

    catCols = [c for c in df.columns if df[c].dtype.name in 'category']
    # trimming surrounding spaces of data strings and making them in lower case
    df = df.applymap(lambda x: x.lower().strip() if isinstance(x, str) else x)
    for col in catCols:
        df[col] = df[col].astype('category').cat.as_ordered()

    for dc in colList:
        df.drop_duplicates(subset=dc, inplace=True)

    return df


@timeit
def dataSanitization(df: pd.DataFrame) -> pd.DataFrame:
    """Data Sanitization"""
    df = df.copy(
    )  # all modifications are applied to the copy of the original Data Frame.

    for task in (task3, task1, task2, task4):
        df = task(df)

    return df


print("Reading input file ...")
start = time.time()
df = pd.read_excel('bank.xlsx')
print(f"Read [bold]{len(df)}[/bold] lines in {timeSince(start)}.\n")
#df= df.rename(columns=str.lower)
sdf = dataSanitization(df)

# using label encoder in order to convert categorical data to a numerical code
le = LabelEncoder()
# Isolataion Forest to capture the anomally of the given data
iforest = IsolationForest(contamination=.01,
                          n_estimators=100,
                          n_jobs=-1,
                          random_state=14)

start = time.time()
features = [
    c for c in sdf.columns
    if sdf[c].dtype.name in ('float64', 'int64', 'category')
]

for c in features:
    if sdf[c].dtype.name == 'category':
        le.fit(sdf[c])
        sdf[c] = le.transform(sdf[c])

iforest.fit(sdf[features])
sdf['score'] = iforest.decision_function(sdf[features])
sdf['anomaly'] = iforest.predict(sdf[features])
print(
    f"Anomaly Detection takes [bold red]{timeSince(start)}[/bold red] with Isolataion Forest Algorithm.\n"
)

prettify(sdf.iloc[np.where(sdf['anomaly'] == -1, True, False)][[
    "Account No", "DATE", "TRANSACTION DETAILS", "VALUE DATE",
    "WITHDRAWAL AMT", "DEPOSIT AMT", "BALANCE AMT"
]])

print(
    f"Total number of Anomaly in data set is {(sdf['anomaly']==-1).sum()} out of {len(df)} records."
)
