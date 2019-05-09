import numpy as np
import matplotlib.pyplot as plt

def readinfo(file):

    with open(file+'_info.txt') as f:
        infolist = f.read()
    info=[]
    for i in infolist.split('\n')[:-1]:
        info.append([float(j) for j in i.split('\t')])

    return info

def statistics(info):
    avg=[]
    std=[]
    num=[]
    for quantity in range(8):
        avg.append(np.average([item[quantity] for item in info]))
        std.append(np.std([item[quantity] for item in info]))
        num.append(len([item[quantity] for item in info]))
    return avg, std, len([item[0] for item in info])    
    
#plt.close("all")

#main_folder='C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'
#folder=main_folder+'19_04_27_B6/export_np/'
##file='meas_diff_04_1'
#
#files=[]
##files=[folder+'meas_diff_{:02.0f}_1'.format(f) for f in [2, 3, 5]]
#files+=[main_folder+'19_04_11_C5/export_np/'+'meas_diff_03_1',
#        main_folder+'19_04_11_C5/export_np/'+'meas_diff_04_1',
#        main_folder+'19_04_17_C3/export_np/'+'meas_diff_02_1',
#        main_folder+'19_04_17_C3/export_np/'+'meas_diff_03_1']
#
#print('no\tx\ty\tcx\tcy\tC\tstd\tint\trelBg')
#for file in files: 
#    info=readinfo(file)
#    info_stat=statistics(info)
#    print('{}\t{:.02f}\t{:.02f}\t{:.02f}\t{:.02f}\t{:.02f}\t{:.05f}\t{:.05f}\t{:.05f}'.format(file[-4:-2], *info_stat[0]))
#    print('n= {}\t{:.02f}\t{:.02f}\t{:.02f}\t{:.02f}\t{:.02f}\t{:.05f}\t{:.05f}\t{:.05f}'.format(info_stat[2], *info_stat[1]))
#    print('------------------------------------------------------------------')
