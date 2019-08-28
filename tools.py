import os

def new_export(folder, name):
    path = folder+name
    
    try:  
        os.mkdir(path)
    except OSError:  
        print ("Creation of the directory %s failed" % path)
    else:  
        print ("Successfully created the directory %s " % path)
        
def all_bin_files(folder):
    out=list()
    for r, d, f in os.walk(folder):
        for file in f:
            if file[-3:]=='bin':
                out.append(file[:-4])
    return out

def SecToMin(sec):
    return '{:.0f}:{:.1f}'.format(sec//60, sec%60)

def frame_times(file_content):
    time0=int(file_content[1].split()[0])
    time_info=[]
    time_last=time0
    for line in file_content[1:]:
        time_actual=int(line.split()[0])
        time_info.append([(time_actual-time0)/1e7, (time_actual-time_last)/1e7])
        time_last=time_actual
    return time_info

def t2i(boo):
    if boo:
        return 1
    else:
        return 0    
    