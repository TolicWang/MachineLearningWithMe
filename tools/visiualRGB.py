def visiualization(color=False):
    """
    可视化
    :param color: 是否彩色
    :return:
    """
    from PIL import Image
    import matplotlib.pyplot as plt
    from sklearn.datasets import fetch_lfw_people
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

if __name__ == '__main__':
    visiualization(color=True)