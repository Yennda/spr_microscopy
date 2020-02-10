from biovideo import BioVideo
import matplotlib.pyplot as plt
import tools as tl
import time as t

yellow='#ffb200'
red='#DD5544'
blue='#0284C0'
black = '#000000'
green = '#009900'

COLORS = [yellow, blue, red, black, green]

time_start=t.time()

tl.clear_all()
plt.close("all")

main_folder='C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'
folder=main_folder+'20_01_30_Tomas_low_concentration_miRNA/'


file = 'raw_01'

video = BioVideo(folder, file, 2)
video.loadData()

for vid in video._videos:
    vid._video['raw'] = vid._video['raw'][:, :, :650]
    vid.refresh()

video.ref_frame = 0


integration = [1, 10, 20, 30, 80]
for i in integration:
    video.make_int(i)
    
    video.spr = True
    video.explore()



fig, axes = plt.subplots()
    
axes.grid(linestyle='--')
axes.set_title('Standard deviation of frames', fontsize=20)
axes.set_xlabel('time [min]', fontsize=20)
axes.set_ylabel('intensity change [a. u.]', fontsize=20)
axes.tick_params(labelsize=20)
axes_std = axes.twinx()
axes_std.tick_params(labelsize=20)
axes_std.set_ylabel('st. dev. [a. u.]', fontsize=20)
end = video.syn_index + len(video._videos[0].video)
for c in video._channels: 
    axes.plot(video.spr_time[video.syn_index:end], video.spr_signals[c][video.syn_index:end], linewidth=1, color=COLORS[c], label='channel {}'.format(c+1))


for i in range(len(integration)):
    axes_std.plot(video.memory[i][0][0], video.memory[i][0][1], linewidth=1, color=COLORS[i], label='{} frames, c. 1'.format(integration[i]))
    axes_std.plot(video.memory[i][1][0], video.memory[i][1][1], linewidth=1, color=COLORS[i], label='{} frames, c. 2'.format(integration[i]), ls=':')
    axes_std.legend(loc=2)

axes.legend(loc=1)
    

print('{:.2f} s'.format(t.time()-time_start))