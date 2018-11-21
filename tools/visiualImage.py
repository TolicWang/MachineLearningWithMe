def visiualization(images, label=[], label_name=[], color=False, row=3, col=5):
    """
    可视化
    :param color: 是否彩色
    :return:
    """
    from PIL import Image
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(row, col)  # 15张图
    for i, axi in enumerate(ax.flat):
        image = images[i]
        if color:
            image = image.transpose(2, 0, 1)
            r = Image.fromarray(image[0]).convert('L')
            g = Image.fromarray(image[1]).convert('L')
            b = Image.fromarray(image[2]).convert('L')
            image = Image.merge("RGB", (r, g, b))
        axi.imshow(image, cmap='bone')
        if len(label_name) > 0:
            axi.set(xticks=[], yticks=[], xlabel=label_name[label[i]])
    plt.show()


if __name__ == '__main__':
    from sklearn.datasets import fetch_lfw_people,load_digits

    faces = fetch_lfw_people(min_faces_per_person=60, color=True)
    visiualization(faces.images, label=faces.target, label_name=faces.target_names, color=True)

    digits = load_digits()
    visiualization(digits.images,label=digits.target, label_name=digits.target_names)
