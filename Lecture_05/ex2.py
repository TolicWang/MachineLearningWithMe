import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV

def feature_selection():
    from sklearn.feature_selection import SelectKBest, f_classif
    import matplotlib.pyplot as plt
    train = pd.read_csv('./data/titanic_train.csv')
    selected_features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    train_x = train[selected_features]
    train_y = train['Survived']
    train_x['Age'].fillna(train_x['Age'].mean(), inplace=True)  # 以均值填充
    train_x['Embarked'].fillna('S', inplace=True)
    train_x.loc[train_x['Embarked'] == 'S', 'Embarked'] = 0
    train_x.loc[train_x['Embarked'] == 'C', 'Embarked'] = 1
    train_x.loc[train_x['Embarked'] == 'Q', 'Embarked'] = 2
    train_x.loc[train_x['Sex'] == 'male', 'Sex'] = 0
    train_x.loc[train_x['Sex'] == 'female', 'Sex'] = 1

    selector = SelectKBest(f_classif, k=5)
    selector.fit(train_x, train_y)
    scores = selector.scores_
    plt.bar(range(len(selected_features)), scores)
    plt.xticks(range(len(selected_features)), selected_features, rotation='vertical')
    plt.show()

    x_train = train_x[['Pclass', 'Sex', 'Fare']]
    y_train = train_y.as_matrix()
    return x_train, y_train
def logistic_regression():
    from sklearn.linear_model import LogisticRegression
    x_train, y_train= feature_selection()
    model = LogisticRegression()
    paras = {'C': np.linspace(0.1, 10, 50)}
    gs = GridSearchCV(model, paras, cv=5, verbose=3)
    gs.fit(x_train, y_train)
    print('best score:', gs.best_score_)
    print('best parameters:', gs.best_params_)


def decision_tree():
    from sklearn.tree import DecisionTreeClassifier
    x_train, y_train = feature_selection()
    model = DecisionTreeClassifier()
    paras = {'criterion': ['gini', 'entropy'], 'max_depth': np.arange(5, 50, 5)}
    gs = GridSearchCV(model, paras, cv=5, verbose=3)
    gs.fit(x_train, y_train)
    print('best score:', gs.best_score_)
    print('best parameters:', gs.best_params_)


def random_forest():
    from sklearn.ensemble import RandomForestClassifier
    x_train, y_train = feature_selection()
    model = RandomForestClassifier()
    paras = {'n_estimators': np.arange(10, 100, 10), 'criterion': ['gini', 'entropy'], 'max_depth': np.arange(5, 50, 5)}
    gs = GridSearchCV(model, paras, cv=5, verbose=3)
    gs.fit(x_train, y_train)
    print('best score:', gs.best_score_)
    print('best parameters:', gs.best_params_)

if __name__ == '__main__':
    # feature_selection()
    # logistic_regression()#0.783
    # decision_tree()#0.814
    random_forest()# 0.814