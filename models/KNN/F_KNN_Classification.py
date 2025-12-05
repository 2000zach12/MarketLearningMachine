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

    dfClean = dfUnclean
    return dfClean

def create_delayed_lagged_features(df, lags, cols,delay):
    for col in cols:
        for i in range(1+delay,lags+1+delay):
            df[f'{col}_lag_{i}'] = df[col].shift(i)

    return df





df = dfCleaner(dfUnclean)
delay = 5
lags = 5
lag_features = ['Close','Volume','Open']
df = create_delayed_lagged_features(df, lags=lags, cols=lag_features,delay=delay)

#Drop Nan
drop=delay+lags
df=df[drop:]

target = 'Close'


Y = df[target]
X = df[[c for c in df.columns if "_lag_" in c]+ [c for c in df.columns if c not in lag_features and c not in ["Date", "Adj Close","High","Low", target]]]

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.20, shuffle=False)




from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsRegressor

knn = KNeighborsRegressor(n_neighbors=1)
knn.fit(X_train_scaled, Y_train)

Y_pred = knn.predict(X_test_scaled)



mae = mean_absolute_error(Y_test, Y_pred)
rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
r2 = r2_score(Y_test, Y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)