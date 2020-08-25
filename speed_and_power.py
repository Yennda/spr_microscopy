import timeit
import numpy as np
import os
import math as m
import numpy as np

from scipy.signal import find_peaks
from scipy.signal import argrelextrema

nm=int(1e4)

'''
nm = 1e4
Find peaks
max
0.22486762399989857
min
0.2603954620001332
------------------------
argrelextrema
max
0.26328810699988026
min
0.2364915769999243
------------------------
FindMaxima
max
1.6727407459998176
min
1.5779822609999883
------------------------
correlation
0.03809590600030788
------------------------

correlation_temporal nm=100
4.291365259000031
improving tri calculation:
3.6510248070001126
    
'''

print('Find peaks')
print('max')
print(timeit.timeit('find_peaks(correlation, height=15)', setup='from scipy.signal import find_peaks; from data import correlation', number=nm))
print('min')
print(timeit.timeit('find_peaks(-correlation, height=15)', setup='from scipy.signal import find_peaks; from data import correlation', number=nm))
print('------------------------')

print('argrelextrema')
print('max')
print(timeit.timeit('argrelextrema(correlation, np.greater)', setup='from scipy.signal import argrelextrema; import numpy as np; from data import correlation', number=nm))
print('min')
print(timeit.timeit('argrelextrema(correlation, np.less)', setup='from scipy.signal import argrelextrema; import numpy as np; from data import correlation', number=nm))
print('------------------------')

#print('FindMaxima')
#print('max')
#print(timeit.timeit('FindMaxima(correlation)', setup='from tools import FindMaxima; import numpy as np; from data import correlation', number=nm))
#print('min')
#print(timeit.timeit('FindMaxima(-correlation)', setup='from tools import FindMaxima; import numpy as np; from data import correlation', number=nm))
#print('------------------------')

print('correlation_temporal')
print(timeit.timeit('correlation_temporal(data_temp, 10, -0.0055)', setup='from image_processing import correlation_temporal; from data import data_temp', number=100))

print('correlation')
print(timeit.timeit('np. correlate(data_temp[150-10:150+10], tri)[0]*1e5', setup='import numpy as np; from data import data_temp, tri', number=nm))


