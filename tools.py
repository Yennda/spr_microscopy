import os
import random as rn


def new_export(folder, name):
    path = folder + name

    try:
        os.mkdir(path)
    except OSError:
        pass
    else:
        print('Directory {} was successfully created.'.format(path))

def before_save_file(path):
    
    if os.path.isfile(path):
        print('=' * 80)
        
        print(
                'Old data in {} will be overwriten.'
                'Type "y" as yes or "n"'
                'as no bellow in the command line.'.format(path))
        
        print('=' * 80)
        result = input()
        
        if result == 'y':
            os.remove(path)
            
            return True
        
        return False
    
    return True

def true_coordinate(x):
    return int((x + 0.5) // 1)

def all_bin_files(folder):
    out = list()
    
    for r, d, f in os.walk(folder):
        
        for file in f:
            
            if file[- 3: ] == 'bin':
                out.append(file[: - 4])
                
    return out

def random_color():
    return (rn.random(), rn.random(), rn.random())

def hex_to_list(color):
    color = color[1:]
    dec_list = [int(color[i: i + 2], 16) for i in range(0, 5, 2)]
    return [d / 256 for d in dec_list]

def SecToMin(sec):
    return '{:.0f}:{:.1f}'.format(sec // 60, sec % 60)

def frame_times(file_content):
    time0 = int(file_content[1].split()[0])
    time_info = []
    time_last = time0
    
    for line in file_content[1:]:
        time_actual = int(line.split()[0])
        time_info.append([
                (time_actual - time0) / 1e7, 
                (time_actual - time_last) / 1e7
                ])
        time_last = time_actual
        
    return time_info


def t2i(boo):
    if boo:
        return 1
    else:
        return 0
    
def clear_all():
    "Clears all the variables from the workspace of the spyder application."
    gl = globals().copy()
    for var in gl:
        if var[0] == '_': continue
        if 'func' in str(globals()[var]): continue
        if 'module' in str(globals()[var]): continue

        del globals()[var]
        
def closest(lst, K):  
    return lst[min(range(len(lst)), key = lambda i: abs(lst[i] - K))] 




