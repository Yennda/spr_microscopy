import numpy as np
import time as t
import sys

from video_processing import Video
import matplotlib.pyplot as plt
import sqlite3
from database_methods import process_data

plt.close("all")
time_start=t.time()

con = sqlite3.connect('database_results.db')
cursor = con.cursor()
   
main_folder='C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'
#folder = '20_04_15_L3/'
#file = 'raw_31_1'

folder = '20_04_15_L3/'
file = 'raw_13_1'

#folder = '20_03_16_K4/'
#file = 'raw_05_1'

folder = '20_02_25_P3/'
file = 'raw_10_1'

#folder = '20_04_20_Q4/'
#file = 'raw_02_1'
#
#folder = '20_03_16_K5/'
#file = 'raw_04_1'

columns = 'AR_TH, EXC_THS, AR_COND, AR_MIN, AR_TH, AR_DIP, AR_NOISE'
data = process_data(
        con.execute("""
SELECT {}
FROM 'masters' as MAS 
INNER JOIN 'experiments' as EXP 
ON MAS.ID = EXP.MASTER_ID
INNER JOIN 'measurements' as MEAS
ON EXP.ID = MEAS.EXPERIMENT_ID 
WHERE FOLDER = '{}' AND FILE = '{}'
;
   """.format(columns, folder, file)),
        columns.split(', ')
        )
con.close()

if len(data) == 0:
    sys.exit()

video = Video(main_folder + folder, file)
video.loadData()

#video._video['raw']=video._video['raw'][100:,600:900,120:]
video._video['raw']=video._video['raw'][100:,400:700,:]
video.make_diff(k = 10)
video.fouriere(level = 20)

"alpha"

video.img_process_alpha(threshold = data['AR_TH'], noise_level = data['AR_NOISE'], dip = data['AR_DIP'])
video.characterize_nps(save = False)
video.exclude_nps(data['EXC_THS'], exclude = True)
video.make_toggle(['diff', 'inta'], [10, 10])

"gamma"
#video.load_idea()
#video.make_corr()
#video._condition = data['AR_COND']
#video._minimal_area = data['AR_MIN']
#
#video.image_process_gamma()  
#video.characterize_nps(save = False)
#video.exclude_nps(data['EXC_THS'], exclude = True)
#video.make_toggle(['diff', 'corr'], [10, 10])

video.characterize_nps(save = False)
video.statistics()

video.explore()
video.histogram('corr')

#video.save_info_measurement(80, 734)

video.info_add('\n--elapsed time--\n{:.2f} s'.format(t.time()-time_start))
print(video.info)