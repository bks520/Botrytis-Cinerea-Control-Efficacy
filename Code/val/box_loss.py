import proplot as pplt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    pplt.rc['font.family'] = 'Times New Roman'

    # 从Excel文件读取第一组数据
    df1 = pd.read_excel('C:\\Users\\zxs\\PycharmProjects\\transtab-main\\exp14results.xlsx')
    # 从Excel文件读取第二组数据
    df2 = pd.read_excel('C:\\Users\\zxs\\PycharmProjects\\transtab-main\\exp28results.xlsx')

    # 每隔10行取样一次数据
    data1 = df1.iloc[::20, :].values
    data2 = df2.iloc[::20, :].values

    pplt.rc.update('subplots', share=False, span=False)
    fig = pplt.figure()

    ax = fig.add_subplot(111)

    # 绘制第一组数据
    ax.plot(np.arange(0, 501, 20), data1[:, 9], linestyle='-', color='b', linewidth=1.5, label='YOLOv8sOBB')
    ax.plot(np.arange(0, 501, 20), data2[:, 9], linestyle='-', color='r', linewidth=1.5, label='Ours')


    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    ax.yaxis.offsetText.set_fontsize(8)


    # 设置 y 轴范围，从 0.1 开始
    ax.set_ylim(0, 0.3)

    ax.format(xlim=(0, 501), xlabel='Epoch', ylabel='val/box_loss')
    # ax.text(0.5, -0.2, '(a)', ha='center', va='center', transform=ax.transAxes)

    ax.legend(ncol=1, bbox_to_anchor=(0.95, 0.95))

    # ax = fig.add_subplot(122)

    # # 绘制第二组数据
    # ax.plot(np.arange(0, 501, 20), data2[:, 1], linestyle='--', color='b', linewidth=1.5, label='Training Loss')
    # ax.plot(np.arange(0, 501, 20), data2[:, 2], linestyle='-', color='r', linewidth=1.5, label='Validation Loss')
    #
    #
    # ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0), useMathText=True)
    # ax.yaxis.offsetText.set_fontsize(8)
    #
    # # 设置 y 轴范围，从 0.1 开始
    # ax.set_ylim(0, 1)
    #
    # ax.format(xlim=(0, 501), xlabel='Epoch', ylabel='Recall')
    # ax.text(0.5, -0.2, '(b)', ha='center', va='center', transform=ax.transAxes)
    #
    # ax.legend(ncol=1, bbox_to_anchor=(0.95, 0.95))

    fig.save('./images/fig126.png')
