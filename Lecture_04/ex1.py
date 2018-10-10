import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report


def load_data():
    from sklearn.datasets import load_iris
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    data = load_iris()
    X = data.data
    y = data.target
    ss = StandardScaler()
    X = ss.fit_transform(X)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    return x_train, y_train, x_test, y_test, data.feature_names


def train():
    x_train, y_train, x_test, y_test, _ = load_data()
    model = DecisionTreeClassifier()
    model.fit(x_train, y_train)
    y_pre = model.predict(x_test)
    print(model.score(x_test, y_test))
    print(classification_report(y_test, y_pre))


def grid_search():
    from sklearn.model_selection import GridSearchCV
    x_train, y_train, x_test, y_test, _ = load_data()
    model = DecisionTreeClassifier()
    parameters = {'max_depth': np.arange(1, 50, 2)}
    gs = GridSearchCV(model, parameters, verbose=5, cv=5)
    gs.fit(x_train, y_train)
    print('最佳模型:', gs.best_params_, gs.best_score_)
    y_pre = gs.predict(x_test)
    print(classification_report(y_test, y_pre))


def tree_visilize():
    from sklearn import tree
    x_train, y_train, x_test, y_test, feature_names = load_data()
    print('类标：', np.unique(y_train))
    print('特征名称：', feature_names)
    model = DecisionTreeClassifier(max_depth=3)
    model.fit(x_train, y_train)
    print(model.score(x_test, y_test))
    with open("allElectronicsData.dot", "w") as f:
        tree.export_graphviz(model, feature_names=feature_names, class_names=['A', 'B', 'C'], out_file=f)


if __name__ == '__main__':
    train()
    # grid_search()
    # tree_visilize()
