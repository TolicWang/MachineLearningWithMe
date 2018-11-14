from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.datasets import fetch_lfw_people
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split,GridSearchCV
import numpy as np
import os


def visiualization(color=False):
    """
    可视化
    :param color: 是否彩色
    :return:
    """
    from PIL import Image
    import matplotlib.pyplot as plt
    faces = fetch_lfw_people(min_faces_per_person=60, color=color)
    fig, ax = plt.subplots(3, 5)  # 15张图
    for i, axi in enumerate(ax.flat):
        image = faces.images[i]
        if color:
            image = image.transpose(2, 0, 1)
            r = Image.fromarray(image[0]).convert('L')
            g = Image.fromarray(image[1]).convert('L')
            b = Image.fromarray(image[2]).convert('L')
            image = Image.merge("RGB", (r, g, b))
        axi.imshow(image, cmap='bone')
        axi.set(xticks=[], yticks=[], xlabel=faces.target_names[faces.target[i]])
    plt.show()


def load_data():

    faces = fetch_lfw_people(min_faces_per_person=60)
    x = faces.images
    x = x.reshape(len(x), -1)
    y = faces.target
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=10)
    return x_train, x_test, y_train, y_test


def model_select():

    x_train, x_test, y_train, y_test = load_data()
    svc = SVC()
    pca = PCA(n_components=20, whiten=True, random_state=42)
    paras = {'svc__C': np.linspace(1, 5, 10), 'svc__gamma': [0.0001, 0.0005, 0.001, 0.005],
             'pca__n_components': np.arange(10, 200, 20)}
    model = make_pipeline(pca, svc)
    gs = GridSearchCV(model,paras,n_jobs=-1,verbose=2)
    gs.fit(x_train, y_train)
    print(gs.best_score_)
    print(gs.best_params_)
    print(gs.best_estimator_)
    print(gs.score(x_test, y_test))
    """
    [Parallel(n_jobs=-1)]: Done 1200 out of 1200 | elapsed:  4.5min finished
    0.8348170128585559
    {'pca__n_components': 90, 'svc__C': 3.2222222222222223, 'svc__gamma': 0.005}
    Pipeline(memory=None,
         steps=[('pca', PCA(copy=True, iterated_power='auto', n_components=90, random_state=42,
      svd_solver='auto', tol=0.0, whiten=True)), ('svc', SVC(C=3.2222222222222223, cache_size=200, class_weight=None, coef0=0.0,
      decision_function_shape='ovr', degree=3, gamma=0.005, kernel='rbf',
      max_iter=-1, probability=False, random_state=None, shrinking=True,
      tol=0.001, verbose=False))])
    0.8605341246290801

    """

if __name__ == "__main__":
    model_select()
