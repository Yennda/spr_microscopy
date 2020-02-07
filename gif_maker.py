from PIL import Image
import glob
import os


main_folder='C:/SPRUP_data_Jenda/2019_03_13_Jenda_microscopy/'

folder=main_folder+'20_01_30_Tomas_low_concentration_miRNA/'


folder += 'exports_bio'



frames = []

#if not os.path.isdir(folder):
#    os.mkdir(folder)


pngs = os.listdir(folder)


#
imgs = [folder + '/' + p for p in pngs]



for i in imgs:
    new_frame = Image.open(i)
    frames.append(new_frame)
 
# Save into a GIF file that loops forever
frames[0].save(folder + '/' + pngs[0][:5] + '.gif', format='GIF',
               append_images=frames[1:],
               save_all=True,
               duration=1000, loop=0)