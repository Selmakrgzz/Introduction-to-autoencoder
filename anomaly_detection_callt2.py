from datasets import load_dataset
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Flatten, Reshape, Input

print("hei")
""" Load the data """
calit2 = load_dataset("PLAN-Lab/mTSBench", "CalIt2")

df_train = calit2["train"].to_pandas()
df_test = calit2["test"].to_pandas()

""" Simple EDA """
# check if there are any missing vals
print(df_train.isna().sum())
# check if the train dataset has anomalies
print(df_train["is_anomaly"].sum()) 
# check the timestamps and how they are spaced in time
ts = pd.to_datetime(df_train["timestamp"])
print(ts.diff().value_counts().head())

""" Scale the data """
WIN = 48 # timestamp every 30 min, 48 30mins in one day
feats = ["in_count", "out_count"] # the two features in the dataset

scaler = MinMaxScaler().fit(np.log1p(df_train[feats]))
x_train = scaler.transform(np.log1p(df_train[feats]))
x_test = scaler.transform(np.log1p(df_test[feats]))

""" Cut into windows """
X_train = sliding_window_view(x_train, WIN, axis=0).transpose(0, 2, 1)
X_test  = sliding_window_view(x_test,  WIN, axis=0).transpose(0, 2, 1)
print(X_train.shape, X_test.shape)

""" Building the model """
model = Sequential()
model.add(Input(shape=(WIN,2)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu')) # bottleneck
model.add(Dense(32, activation='relu'))
model.add(Dense(WIN*2, activation='sigmoid'))
model.add(Reshape((WIN,2)))

model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()
model.fit(X_train, X_train, epochs=100, batch_size=32, validation_split=0.1, verbose=1)

""" Reconstruction per window """
recon_test = model.predict(X_test)
win_err = np.mean((recon_test - X_test) ** 2, axis=(1,2))

""" Spread window errors to every time step """
point_err = np.convolve(win_err, np.ones(WIN)) / np.convolve(np.ones(len(win_err)), np.ones(WIN))

""" Threshold from normal errors """
recon_train = model.predict(X_train)
train_err = np.mean((recon_train - X_train) ** 2, axis=(1, 2))
threshold = np.percentile(train_err, 99)
pred_anomaly = point_err > threshold

""" Evaluate against the labels """
labels = df_test["is_anomaly"].values
print("ROC-AUC:", roc_auc_score(labels, point_err))