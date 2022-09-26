from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText


def main():

    epoch = []

    train_total = []
    train_img = []
    train_xy = []
    train_dis = []

    valid_total = []
    valid_img = []
    valid_xy = []
    valid_dis = []

    test_total = []
    test_img = []
    test_xy = []
    test_dis = []

    with open(file_path, 'r') as log:
        for line in log.readlines()[start_line: end_line]:
            if not line.startswith('epoch'):
                continue
            line = line.strip().split(',')

            epoch.append(int(line[0].split()[-1]))

            train_total.append(8*float(line[1].split()[1]))
            train_img.append(8*float(line[1].split()[2]))
            train_xy.append(8*float(line[1].split()[3]))
            train_dis.append(float(line[1].split()[4]))

            valid_total.append(8*float(line[2].split()[1]))
            valid_img.append(8*float(line[2].split()[2]))
            valid_xy.append(8*float(line[2].split()[3]))
            valid_dis.append(float(line[2].split()[4]))

            test_total.append(8*float(line[3].split()[1]))
            test_img.append(8*float(line[3].split()[2]))
            test_xy.append(8*float(line[3].split()[3]))
            test_dis.append(float(line[3].split()[4]))

    print(epoch)

    fig = plt.figure(figsize=[24, 18])
    # plt.title('Batch Size {}, t size {}, img loss weight {}, \n lr {} aug {}'.format(8, 16, 2e-5, 1e-4, 'Pre-Aug'),
    #           fontsize=20)

    ax = fig.add_subplot(2, 2, 1)
    ax.plot(epoch, train_total, 'b-', linewidth=2, label='train')
    ax.plot(epoch, valid_total, 'r-', linewidth=2, label='valid')
    ax.plot(epoch, test_total, 'g-', linewidth=2, label='test')
    ax.set_xlabel('Epoch', fontsize=30)
    ax.set_ylabel('Loss', fontsize=30)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.legend(loc='upper right', prop={'size': 20})
    ax.grid()
    ax.text(.5, .9, 'Total Weighted Loss', horizontalalignment='center', transform=ax.transAxes, fontsize=20)

    ax = fig.add_subplot(2, 2, 3)
    ax.plot(epoch, train_img, 'b-', linewidth=2, label='train')
    ax.plot(epoch, valid_img, 'r-', linewidth=2, label='valid')
    ax.plot(epoch, test_img, 'g-', linewidth=2, label='test')
    ax.set_xlabel('Epoch', fontsize=30)
    ax.set_ylabel('Img_Loss', fontsize=30)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.grid()
    ax.legend(loc='upper right', prop={'size': 20})
    ax.text(.5, .9, 'Image Loss', horizontalalignment='center', transform=ax.transAxes, fontsize=20)

    ax = fig.add_subplot(2, 2, 2)
    ax.plot(epoch, train_xy, 'b-', linewidth=2, label='train')
    ax.plot(epoch, valid_xy, 'r-', linewidth=2, label='valid')
    ax.plot(epoch, test_xy, 'g-', linewidth=2, label='test')
    ax.set_xlabel('Epoch', fontsize=30)
    ax.set_ylabel('XY_Loss', fontsize=30)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.grid()
    ax.legend(loc='upper right', prop={'size': 20})
    ax.text(.5, .9, 'Point Loss', horizontalalignment='center', transform=ax.transAxes, fontsize=20)

    ax = fig.add_subplot(2, 2, 4)
    ax.plot(epoch, train_dis, 'b-', linewidth=2, label='train')
    ax.plot(epoch, valid_dis, 'r-', linewidth=2, label='valid')
    ax.plot(epoch, test_dis, 'g-', linewidth=2, label='test')
    ax.set_xlabel('Epoch', fontsize=30)
    ax.set_ylabel('Point Distance', fontsize=30)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.grid()
    ax.legend(loc='upper right', prop={'size': 20})
    # anchored_text = AnchoredText('Batch Size {}, t size {}, img loss weight {}, \n'
    #                              ' lr {} aug {}'.format(8, 16, 2e-5, 1e-4, 'Pre-Aug'),
    #                              loc='upper center', prop=dict(fontweight='normal', size=12)
    #                              )
    # ax.add_artist(anchored_text)
    # plt.show()

    plt.show()


if __name__ == '__main__':
    file_dir = '/home/liuwei/Angio/Models/MainCurve_LAO/'
    file_name = 'model2_C32D2O5I5FL0_6_log_mod'
    file_path = file_dir + file_name
    start_line = 0
    end_line = None
    main()
