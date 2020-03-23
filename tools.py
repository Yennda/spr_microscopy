import os
import matplotlib.pyplot as plt
import numpy as np
import random as rn
from tkinter import messagebox

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
        print("Directory %s was successfully created." % path)

def before_save_file(path):
    if os.path.isfile(path):
        print('='*80)
        print('Old data in {} will be overwriten. Type "y" as yes or "n" as no bellow in the command line.'.format(path))
        print('='*80)
        result = input()
        if result == 'y':
            return True
        return False
    return True

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

def random_color():
    return (rn.random(), rn.random(), rn.random())



