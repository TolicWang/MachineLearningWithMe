import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


def load_and_analyse_data():
    data = pd.read_csv('./data/creditcard.csv')

    # ----------------------查看样本分布情况----------------------------------
    # count_classes = pd.value_counts(data['Class'],sort=True).sort_index()
    # print(count_classes)# negative 0 :284315   positive 1 :492
    # count_classes.plot(kind='bar')
    # plt.title('Fraud class histogram')
    # plt.xlabel('Class')
    # plt.ylabel('Frequency')
    # plt.show()
    # --------------------------------------------------------------------------

    # ----------------------预处理---------------------------------------------

    # ----------------------标准化Amount列---------
    data['normAmout'] = StandardScaler().fit_transform(data['Amount'].values.reshape(-1, 1))
    data = data.drop(['Time', 'Amount'], axis=1)
    # ----------------------------------------------

    X = data.ix[:, data.columns != 'Class']
    y = data.ix[:, data.columns == 'Class']
    positive_number = len(y[y.Class == 1])  # 492
    negative_number = len(y[y.Class == 0])  # 284315
    positive_indices = np.array(y[y.Class == 1].index)
    negative_indices = np.array(y[y.Class == 0].index)

    # ----------------------采样-------------------
    random_negative_indices = np.random.choice(negative_indices, positive_number, replace=False)
    random_negative_indices = np.array(random_negative_indices)
    under_sample_indices = np.concatenate([positive_indices, random_negative_indices])
    under_sample_data = data.iloc[under_sample_indices, :]
    X_sample = under_sample_data.ix[:, under_sample_data.columns != 'Class']
    y_sample = under_sample_data.ix[:, under_sample_data.columns == 'Class']
    return np.array(X), np.array(y).reshape(len(y)), np.array(X_sample), np.array(y_sample).reshape(len(y_sample))


if __name__ == '__main__':
    X, y, X_sample, y_sample = load_and_analyse_data()
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.3, random_state=30)
    X_train, X_dev, y_train, y_dev = train_test_split(X_sample, y_sample, test_size=0.3,
                                                                                    random_state=1)

    print("X_train:{}  X_dev:{}  X_test:{}".format(len(y_train),len(y_dev),len(y_test)))
    model = LogisticRegression()
    parameters = {'C': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]}
    gs = GridSearchCV(model, parameters, verbose=5, cv=5)
    gs.fit(X_train, y_train)
    print('最佳模型:', gs.best_params_, gs.best_score_)
    print('在采样数据上的性能表现：')
    print(gs.score(X_dev, y_dev))
    y_dev_pre = gs.predict(X_dev)
    print(classification_report(y_dev, y_dev_pre))
    print('在原始数据上的性能表现：')
    print(gs.score(X_test, y_test))
    y_pre = gs.predict(X_test)
    print(classification_report(y_test, y_pre))
