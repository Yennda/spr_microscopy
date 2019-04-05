from video_processing import Video

import matplotlib.pyplot as plt
import tools

plt.close("all")

folder='C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/19_03_14_C5/'
tools.new_export(folder, 'export')
files=tools.all_bin_files(folder)



i=0
for file in files:
    print('{} / {}'.format(i, len(files)))
    video=Video(folder,file)
    video.loadData()
    video.rng=[-0.01, 0.01]
    if file.find('diff')!=-1:
        video.export(folder+'export/'+file, auto=False)
    else:
        video.export(folder+'export/'+file)
    i+=1


