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

lag_features = ['Close','Volume','Open']
df = create_lagged_features(df, lags=4, cols=lag_features)

#Drop Nan
df=df[8:]

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

from sklearn.neighbors import KNeighborsRegressor
# Fitting KNN regression to the Training set

KNN_regression = KNeighborsRegressor(n_neighbors=1)
KNN_regression.fit(X_train, Y_train)

# Predicting the Test set results
Y_hat = KNN_regression.predict(X_test)

sns.scatterplot(x=Y_test, y=Y_hat, alpha=0.6)
sns.lineplot(Y_test)


print(KNN_regression.score(X_train, Y_train))

print(X_train.shape)


plt.xlabel('Date', fontsize=14)
plt.ylabel('Price', fontsize=14)
plt.title('Actual vs Predicted  wage (test set)', fontsize=17)
plt.show()


from sklearn.model_selection import cross_val_score

RMSE_CV = []
RMSE_test = []

k = 40

for i in range(1, k):
    KNN_i = KNeighborsRegressor(n_neighbors=i)
    KNN_i.fit(X_train, Y_train)
    RMSE_i = np.sqrt(
        np.mean(-1 * cross_val_score(estimator=KNN_i, X=X_train, y=Y_train, cv=10, scoring="neg_mean_squared_error")))
    RMSE_CV.append(RMSE_i)

    RMSE_test.append(np.sqrt(np.mean(np.square(Y_test - KNN_i.predict(X_test)))))

optimal_k = pd.DataFrame({'RMSE_CV': np.round(RMSE_CV, 2), 'RMSE_test': np.round(RMSE_test, 2), 'K': range(1, k)})


print(optimal_k.head(40))


