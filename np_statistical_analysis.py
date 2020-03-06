import numpy as np
import matplotlib.pyplot as plt


def readinfo(file):
    with open(file + '_info.txt') as f:
        infolist = f.read()
    info = []
    for i in infolist.split('\n')[:-1]:
        info.append([float(j) for j in i.split('\t')])

    return info


def statistics(info):
    avg = []
    std = []
    num = []
    for quantity in range(7):
        avg.append(np.average([item[quantity] for item in info]))
        std.append(np.std([item[quantity] for item in info]))
        num.append(len([item[quantity] for item in info]))
    return avg, std, len([item[0] for item in info])

plt.close("all")

main_folder='C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'
folder=main_folder+'20_02_25_P3/'




if __name__ == "__main__":
    files=[folder + 'export_np/' + 'raw_{:02.0f}_1'.format(f) for f in   [10]]
#    files=[folder+'meas_raw_{:02.0f}_1'.format(f) for f in   [4]]


#sizes + [contrast, int_np_norm, int_np_norm_px, int_bg_px, std]
#    print('no\tx\ty\tcx\tcy\tC\tstd\tint\tmaxint\trelBg')
    print('x \ty \tcon \tI_n \tI_n px \tI_b px \tstd')
    for file in files:
       info=readinfo(file)
       info_stat=statistics(info)
       print('{}\t{:.02f}\t{:.02f}\t{:.02f}\t{:.02f}\t{:.02f}\t{:.05f}\t{:.05f}'.format(file[-4:-2], *info_stat[0]))
       print('n= {}\t{:.02f}\t{:.02f}\t{:.02f}\t{:.02f}\t{:.02f}\t{:.05f}\t{:.05f}'.format(info_stat[2], *info_stat[1]))
       print('------------------------------------------------------------------')
