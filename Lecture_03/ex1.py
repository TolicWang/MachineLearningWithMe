import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split

def load_and_analyse_data(sampling=None):
    '''
    本函数的用作时分析数据并进行对应的预处理
    :param sampling: 表示采用的方式，默认为不采样。 'under'表示下采样，'over'表示过采样
    :return:
    '''
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
    y = data.ix[:, data.columns =='Class']
    positive_number = len(y[y.Class==1])# 492
    negative_number = len(y[y.Class==0])# 284315
    positive_indices = np.array(y[y.Class==1].index)
    negative_indices = np.array(y[y.Class==0].index)

    # ----------------------采样-------------------
    if sampling == 'under':
        random_negative_indices = np.random.choice(negative_indices,positive_number,replace=False)
        random_negative_indices = np.array(random_negative_indices)
        under_sample_indices = np.concatenate([positive_indices,random_negative_indices])
        under_sample_data = data.iloc[under_sample_indices,:]
        X = under_sample_data.ix[:,under_sample_data.columns != 'Class']
        y = under_sample_data.ix[:,under_sample_data.columns == 'Class']
        print('Percentage of positive tranctions:',len(under_sample_data[under_sample_data.Class==0])/len(under_sample_indices))
        print('Percentage of negative tranctions:',len(under_sample_data[under_sample_data.Class==1])/len(under_sample_indices))
    elif sampling == 'over':
        pass
    else:
        pass
    return np.array(X),np.array(y)

if __name__ == '__main__':
    X,y=load_and_analyse_data(sampling='under')
    X_train,X_test,y_train,y_test = train_test_split()
