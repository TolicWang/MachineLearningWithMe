from ex1 import load_data_and_preprocessing
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def stacking():
    s = 0
    x_train, y_train, x_test = load_data_and_preprocessing()
    kf = KFold(n_splits=5)
    rfc = RandomForestClassifier(criterion='entropy', max_depth=10, n_estimators=60)
    gbc = GradientBoostingClassifier(learning_rate=0.1, max_depth=5, n_estimators=90)
    for train_index, test_index in kf.split(x_train):
        train_x, test_x = x_train[train_index], x_train[test_index]
        train_y, test_y = y_train[train_index], y_train[test_index]
        rfc.fit(train_x, train_y)
        rfc_pre = rfc.predict_proba(test_x)[:,1]
        gbc.fit(train_x, train_y)
        gbc_pre = gbc.predict_proba(test_x)[:,1]
        y_pre = ((rfc_pre+gbc_pre)/2 >= 0.5)*1
        acc = sum((test_y == y_pre)*1)/len(y_pre)
        s += acc
        print(acc)
    print('Accuracy: ',s/5)# 0.823


if __name__ == '__main__':
    stacking()
