import os
import matplotlib.pyplot as plt
import numpy as np

yellow='#ffb200'
red='#DD5544'
blue='#0284C0'
black='#000000'
green='#008000'

COLORS = [yellow, blue, red, black, green]
SIDES = ['left', 'right', 'bottom', 'top']


def new_export(folder, name):
    path = folder + name

    try:
        os.mkdir(path)
    except OSError:
#        print("Creation of the directory %s failed" % path)
        pass
    else:
        print("Successfully created the directory %s " % path)


def all_bin_files(folder):
    out = list()
    for r, d, f in os.walk(folder):
        for file in f:
            if file[-3:] == 'bin':
                out.append(file[:-4])
    return out


def SecToMin(sec):
    return '{:.0f}:{:.1f}'.format(sec // 60, sec % 60)


def frame_times(file_content):
    time0 = int(file_content[1].split()[0])
    time_info = []
    time_last = time0
    for line in file_content[1:]:
        time_actual = int(line.split()[0])
        time_info.append([(time_actual - time0) / 1e7, (time_actual - time_last) / 1e7])
        time_last = time_actual
    return time_info


def t2i(boo):
    if boo:
        return 1
    else:
        return 0
    
def clear_all():
    """Clears all the variables from the workspace of the spyder application."""
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]
        
def closest(lst, K): 
      
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))] 

def FindMaxima(numbers):
    maxima = []
    length = len(numbers)
 
#    for i in range(1, length-1):     
    i = 0
    while i<(length-1):
        if numbers[i] > numbers[i-1] and numbers[i] > numbers[i+1]:
            maxima.append(numbers[i])
            i+=10
        else:
            i+=1
    
#    if numbers[length-1] > numbers[length-2]:    
#        maxima.append(numbers[length-1])        
    return maxima


