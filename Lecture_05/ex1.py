import pandas as pd
from sklearn.model_selection import GridSearchCV
import numpy as np


def load_data_and_preprocessing():
    train = pd.read_csv('./data/titanic_train.csv')
    test = pd.read_csv('./data/test.csv')
    # print(train['Name'])
    # print(titannic_train.describe())
    # print(train.info())
    train_y = train['Survived']
    selected_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    train_x = train[selected_features]
    train_x['Age'].fillna(train_x['Age'].mean(), inplace=True)  # 以均值填充
    # print(train_x['Embarked'].value_counts())
    train_x['Embarked'].fillna('S', inplace=True)
    # print(train_x.info())

    test_x = test[selected_features]
    test_x['Age'].fillna(test_x['Age'].mean(), inplace=True)
    test_x['Fare'].fillna(test_x['Fare'].mean(), inplace=True)
    # print(test_x.info())

    train_x.loc[train_x['Embarked'] == 'S', 'Embarked'] = 0
    train_x.loc[train_x['Embarked'] == 'C', 'Embarked'] = 1
    train_x.loc[train_x['Embarked'] == 'Q', 'Embarked'] = 2
    train_x.loc[train_x['Sex'] == 'male', 'Sex'] = 0
    train_x.loc[train_x['Sex'] == 'female', 'Sex'] = 1
    x_train = train_x.as_matrix()
    y_train = train_y.as_matrix()

    test_x.loc[test_x['Embarked'] == 'S', 'Embarked'] = 0
    test_x.loc[test_x['Embarked'] == 'C', 'Embarked'] = 1
    test_x.loc[test_x['Embarked'] == 'Q', 'Embarked'] = 2
    test_x.loc[test_x['Sex'] == 'male', 'Sex'] = 0
    test_x.loc[test_x['Sex'] == 'female', 'Sex'] = 1
    x_test = test_x
    return x_train, y_train, x_test


def logistic_regression():
    from sklearn.linear_model import LogisticRegression
    x_train, y_train, x_test = load_data_and_preprocessing()
    model = LogisticRegression()
    paras = {'C': np.linspace(0.1, 10, 50)}
    gs = GridSearchCV(model, paras, cv=5, verbose=3)
    gs.fit(x_train, y_train)
    print('best score:', gs.best_score_)
    print('best parameters:', gs.best_params_)


def decision_tree():
    from sklearn.tree import DecisionTreeClassifier
    x_train, y_train, x_test = load_data_and_preprocessing()
    model = DecisionTreeClassifier()
    paras = {'criterion': ['gini', 'entropy'], 'max_depth': np.arange(5, 50, 5)}
    gs = GridSearchCV(model, paras, cv=5, verbose=3)
    gs.fit(x_train, y_train)
    print('best score:', gs.best_score_)
    print('best parameters:', gs.best_params_)


def random_forest():
    from sklearn.ensemble import RandomForestClassifier
    x_train, y_train, x_test = load_data_and_preprocessing()
    model = RandomForestClassifier()
    paras = {'n_estimators': np.arange(10, 100, 10), 'criterion': ['gini', 'entropy'], 'max_depth': np.arange(5, 50, 5)}
    gs = GridSearchCV(model, paras, cv=5, verbose=3)
    gs.fit(x_train, y_train)
    print('best score:', gs.best_score_)
    print('best parameters:', gs.best_params_)


def gradient_boosting():
    from sklearn.ensemble import GradientBoostingClassifier
    x_train, y_train, x_test = load_data_and_preprocessing()
    model = GradientBoostingClassifier()
    paras = {'learning_rate': np.arange(0.1, 1, 0.1), 'n_estimators': range(80, 120, 10), 'max_depth': range(5, 10, 1)}
    gs = GridSearchCV(model, paras, cv=5, verbose=3,n_jobs=2)
    gs.fit(x_train, y_train)
    print('best score:', gs.best_score_)
    print('best parameters:', gs.best_params_)


if __name__ == '__main__':
    # logistic_regression()  # 0.7979
    # decision_tree()#0.813
    # random_forest()  # 0.836  {'criterion': 'entropy', 'max_depth': 10, 'n_estimators': 60}
    gradient_boosting()#0.830  {'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 90}
