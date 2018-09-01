from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
import numpy as np
%matplotlib inline
from sklearn.datasets import load_digits
digit=load_digits()
x=(digit.data)
y=(digit.target)
from sklearn.preprocessing import StandardScaler
x_Std=StandardScaler().fit_transform(x)
x_Std
from sklearn.decomposition import PCA
pca = PCA(n_components=40)
principalComponents = pca.fit_transform(x_Std)
principalDf = pd.DataFrame(data = principalComponents)
principalDf
pca.explained_variance_ratio_
from sklearn.model_selection import train_test_split
principalDf_train,principalDf_test,y_train,y_test = train_test_split(principalDf,y,test_size = 0.3)
model=LogisticRegression()
model.fit(principalDf_train,y_train)
pred=model.predict(principalDf_test)
pred
from sklearn import metrics
metrics.accuracy_score(y_test,pred)
from sklearn.metrics import classification_report
print(classification_report(y_test,pred))
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test,pred)
