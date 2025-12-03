import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error


sns.set()  #if you want to use seaborn themes with matplotlib functions
import statsmodels.api as sm

import warnings

warnings.filterwarnings('ignore')

dfUnclean=pd.read_excel('F 1021 to 1025.xlsx')

def dfCleaner(dfUnclean):
    # Identify dividend rows
    div_mask = dfUnclean["Open"].astype(str).str.contains("Dividend", case=False, na=False)

    div_rows = dfUnclean[div_mask]
    price_rows = dfUnclean[~div_mask]

    # Extract dividend amount
    div_rows["Dividend"] = div_rows["Open"].astype(str).str.extract(r"([0-9]*\.?[0-9]+)").astype(float)
    div_rows = div_rows[["Date", "Dividend"]]

    # Convert Date to datet
    div_rows["Date"] = pd.to_datetime(div_rows["Date"])
    price_rows["Date"] = pd.to_datetime(price_rows["Date"])

    dfUnclean = price_rows.merge(div_rows, on="Date", how="left")

    if (dfUnclean["Dividend"].isna().any() == True): dfUnclean["Dividend"] = dfUnclean["Dividend"].fillna(0)

    # convert volume to int
    dfUnclean["Volume"] = dfUnclean["Volume"].astype(int)

    df = dfUnclean
    return df

def create_lagged_features(df, lags, cols):
    for col in cols:
        for i in range(1,lags+1):
            df[f'{col}_lag_{i}'] = df[col].shift(i)

    return df





df = dfCleaner(dfUnclean)
