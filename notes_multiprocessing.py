from multiprocessing import Pool

import time

work = [i for i in range(4)]



def work_log(work_data):
    return work_data**2

def pool_handler():
    p = Pool(4)
    out = p.map(work_log, work)
    print(out)


if __name__ == '__main__':
    memory = []
    pool_handler()
