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
    for quantity in range(9):
        avg.append(np.average([item[quantity] for item in info]))
        std.append(np.std([item[quantity] for item in info]))
        num.append(len([item[quantity] for item in info]))
    return avg, std, len([item[0] for item in info])

plt.close("all")

main_folder='C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'
folder=main_folder+'19_08_29_L3/export_np/'
#
#main_folder='C:/Users/jabuk/Documents/jaderka/ufe/results/L3/'
#folder=main_folder
##file='meas_diff_04_1'
#
files=[]
#files=[folder+'norm_{:02.0f}_1'.format(f) for f in [i for i in range(1,15+1)]+[24, 23, 22, 25, 26]]
files+=[main_folder+'19_08_29_L3/export_np/'+'norm_32_10fps',
        main_folder+'19_08_29_L3/export_np/'+'norm_32_5fps',
        main_folder+'19_08_29_L3/export_np/'+'norm_32_2fps',
        main_folder+'19_08_29_L3/export_np/'+'norm_32_1fps',]

print('no\tx\ty\tcx\tcy\tC\tstd\tint\tmaxint\trelBg')
for file in files:
   info=readinfo(file)
   info_stat=statistics(info)
   print('{}\t{:.02f}\t{:.02f}\t{:.02f}\t{:.02f}\t{:.02f}\t{:.05f}\t{:.05f}\t{:.05f}\t{:.05f}'.format(file[-4:-2], *info_stat[0]))
   print('n= {}\t{:.02f}\t{:.02f}\t{:.02f}\t{:.02f}\t{:.02f}\t{:.05f}\t{:.05f}\t{:.05f}\t{:.05f}'.format(info_stat[2], *info_stat[1]))
   print('------------------------------------------------------------------')

print('[')
for file in files:
   info=readinfo(file)
   info_stat=statistics(info)
   print('{},'.format(info_stat[0]))
   
print(']')