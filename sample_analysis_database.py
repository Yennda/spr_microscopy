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
folder = '20_04_20_Q4/'
file = 'raw_28_1'

#folder = '20_04_15_L3/'
#file = 'raw_07_1'

columns = 'AR_TH, EXC_THS, AR_COND, AR_MIN'
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

video._video['raw']=video._video['raw'][100:,430:730,1:]
video.make_diff(k = 10)
video.fouriere(level = 20)


"gamma"
video.load_idea()
video.make_corr()
video._condition = data['AR_COND']
video._minimal_area = data['AR_MIN']

video.image_process_gamma(data['AR_TH'])  
video.characterize_nps(save = False)
video.exclude_nps(data['EXC_THS'], exclude = True)
video.make_toggle(['diff', 'corr'], [10, 10])

video.statistics()

video.explore()
video.histogram('corr')

#video.save_info_measurement(80, 734)

video.info_add('\n--elapsed time--\n{:.2f} s'.format(t.time()-time_start))
print(video.info)