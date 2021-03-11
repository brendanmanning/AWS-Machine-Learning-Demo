import pandas as pd

from sklearn.model_selection import train_test_split

df = pd.read_csv('./stroke_dataset_train.csv')

Y = df['stroke']
X = df.loc[:, df.columns != 'stroke']


XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.33)
