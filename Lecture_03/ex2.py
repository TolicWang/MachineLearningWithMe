import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

def load_and_analyse_data():
    data = pd.read_csv('./data/creditcard.csv')
    # ----------------------预处理---------------------------------------------

    # ----------------------标准化Amount列---------
    data['normAmout'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
    data = data.drop(['Time', 'Amount'], axis=1)
    # ----------------------------------------------

    X = data.ix[:, data.columns != 'Class']
    y = data.ix[:, data.columns == 'Class']
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)
    # ----------------------采样-------------------
    sample_solver = SMOTE(random_state=0)
    X_sample ,y_sample = sample_solver.fit_sample(X_train,y_train)
    return np.array(X_test),np.array(y_test).reshape(len(y_test)),np.array(X_sample),np.array(y_sample).reshape(len(y_sample))

if __name__ == '__main__':
    X_test, y_test, X_sample, y_sample  = load_and_analyse_data()
    X_train,X_dev,y_train,y_dev = train_test_split(X_sample,y_sample,test_size=0.3,random_state=1)

    print("X_train:{}  X_dev:{}  X_test:{}".format(len(y_train), len(y_dev), len(y_test)))
    model = LogisticRegression()
    parameters = {'C':[0.001,0.003,0.01,0.03,0.1,0.3,1,3,10]}
    gs  = GridSearchCV(model,parameters,verbose=5,cv=5)
    gs.fit(X_train,y_train)
    print('最佳模型:',gs.best_params_,gs.best_score_)
    print('在采样数据上的性能表现：')
    print(gs.score(X_dev,y_dev))
    y_dev_pre = gs.predict(X_dev)
    print(classification_report(y_dev,y_dev_pre))
    print('在原始数据上的性能表现：')
    print(gs.score(X_test,y_test))
    y_pre = gs.predict(X_test)
    print(classification_report(y_test,y_pre))
