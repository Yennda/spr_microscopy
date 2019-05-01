import numpy as np
import matplotlib.pyplot as plt

def readinfo(folder, file):

    with open(folder+file+'_info.txt') as f:
        infolist = f.read()
    info=[]
    for i in infolist.split('\n')[:-1]:
        info.append([float(j) for j in i.split('\t')])

    return info

def statistics(info):
    avg=[]
    std=[]
    num=[]
    for quantity in range(6):
        avg.append(np.average([item[quantity] for item in info]))
        std.append(np.std([item[quantity] for item in info]))
        num.append(len([item[quantity] for item in info]))
    return avg, std, num    
    
plt.close("all")

main_folder='C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'
folder=main_folder+'19_04_27_B6/export_np/'

file='meas_diff_04_1'

files=[2, 4]

for f in files: 
    info=readinfo(folder, 'meas_diff_{:02.0f}_1'.format(f))
    info_stat=statistics(info)
    print(info_stat)
