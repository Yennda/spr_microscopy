from multiprocessing import Pool

import time as t
import math as m
work = [i for i in range(2000)]



def work_log(work_data):
    return sum([1/m.factorial(w) for w in range(work_data)])

def pool_handler():
    p = Pool(1000)
    out = p.map(work_log, work)
    p.close()
    print(out)


if __name__ == '__main__':
    
    time_start=t.time()
    memory = []
    pool_handler()

#    print([work_log(w) for w in work])

    print('ELAPSED TIME: {:.2f} s'.format(t.time()-time_start))