import numpy as np
from tools import readinfo, statistics




main_folder='C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'
folder=main_folder+'20_02_25_P3/'
#folder=main_folder+'20_02_26_Q3/'
#folder=main_folder+'20_02_26_L3/'
#folder=main_folder+'20_03_16_K4/'

if __name__ == "__main__":
    files = [
            folder + 
            'exports_np/' + 
            'raw_{:02.0f}_1'.format(f) for f in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
#            'raw_{:02.0f}_1'.format(f) for f in [4, 5, 6, 7, 8, 9, 10, 18, 21]
#            'raw_{:02.0f}_1'.format(f) for f in [3, 4, 5, 6, 7, 8]
            ]

    print('\tx \ty \tcon \tI_np \tI_np px\tI_bg px\tstd')
    
    for file in files:
       info = readinfo(file)
       info_stat = statistics(info)
       print('{}\t{:.02f}\t{:.02f}\t{:.02f}\t{:.02f}\t{:.02f}\t{:.05f}\t{:.05f}'.format(file[-4:-2], *info_stat[0]))
       print('n= {}\t{:.02f}\t{:.02f}\t{:.02f}\t{:.02f}\t{:.02f}\t{:.05f}\t{:.05f}'.format(info_stat[2], *info_stat[1]))
       print('------------------------------------------------------------------')

