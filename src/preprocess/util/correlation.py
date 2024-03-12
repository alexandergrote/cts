"""
Correlation measures for categorical data.

Taken from:
    - https://towardsdatascience.com/the-search-for-categorical-correlation-a1cf7f1888c9
    - https://www.kaggle.com/code/shakedzy/alone-in-the-woods-using-theil-s-u-for-survival

"""

import math
import scipy.stats as ss
from collections import Counter



def conditional_entropy(x,y):
    # entropy of x given y
    y_counter = Counter(y)
    xy_counter = Counter(list(zip(x,y)))
    total_occurrences = sum(y_counter.values())
    entropy = 0
    for xy in xy_counter.keys():
        p_xy = xy_counter[xy] / total_occurrences
        p_y = y_counter[xy[1]] / total_occurrences
        entropy += p_xy * math.log(p_y/p_xy)
    return entropy


def theils_u(x, y):
    s_xy = conditional_entropy(x,y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n/total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x
    

if __name__ == '__main__':

    x = [1, 1, 1, 2, 2, 2, 3, 3, 3]
    y = [1, 2, 2, 1, 1, 1, 2, 2, 1]

    print(theils_u(x, y))
    print(theils_u(y, x))
    print(theils_u(x, x))
    print(theils_u(y, y))
    print(theils_u(x, [1, 1, 1, 1, 1, 1, 1, 1, 1]))
    print(theils_u(x, [1, 2, 3, 1, 2, 3, 1, 2, 3]))
    print(theils_u(x, [3, 3, 3, 3, 3, 3, 3, 3, 3]))
    print(theils_u([1, 2, 3, 3], [3, 2, 1, 3]))
    print(theils_u([1, 2, 3], [3, 2, 1]))