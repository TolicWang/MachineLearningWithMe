def pie_chart(data = {'第一部分': 55, '第二部分': 35, '第三部分': 10}):
    """
    本函数的作用是绘制饼状图
    :param data: 接受的输入data必须为字典类型，如 data={'第一部分':55, '第二部分':35, '第三部分':10}
    :return:
    """
    import matplotlib.pyplot as plt
    import matplotlib

    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False
    label_list = list(data.keys())  # 各部分标签
    size = list(data.values())# 各部分大小
    # color = ["red", "green", "blue"]  # 各部分颜色
    explode = [0]*len(size)
    explode[size.index(max(size))] = 0.05# 值最大的部分突出
    # """
    # 绘制饼图
    # explode：设置各部分突出
    # label: 设置各部分标签
    # labeldistance: 设置标签文本距圆心位置，1.1表示1.1倍半径
    # autopct：设置圆里面文本
    # shadow：设置是否有阴影
    # startangle：起始角度，默认从0开始逆时针转
    # pctdistance：设置圆内文本距圆心距离
    #
    # 返回值
    # l_text：圆内部文本，matplotlib.text.Text
    # object
    # p_text：圆外部文本
    # """
    patches, l_text, p_text = plt.pie(size, explode=explode, labels=label_list,
                                      labeldistance=1.1, autopct="%1.1f%%", shadow=False, startangle=90,
                                      pctdistance=0.8)# color

    plt.axis("equal")  # 设置横轴和纵轴大小相等，这样饼才是圆的
    plt.legend(loc='lower left')# 标签显示角落
    plt.show()
if __name__ =="__main__":
    pie_chart()