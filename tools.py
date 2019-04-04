
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

