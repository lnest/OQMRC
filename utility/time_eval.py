# -*- coding: utf-8 -*-
# ------------------------------------
# Create On 2018/9/18
# File Name: time_eval
# Edit Author: lnest
# ------------------------------------

import time
from functools import wraps


def time_count(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        s_time = time.time()
        data = func(*args, **kwargs)
        e_time = time.time()
        print('run {} cost: {}'.format(func.__name__, e_time - s_time))
        return data
    return wrapper


class TimeCountBlock:
    def __init__(self, name=None):
        self._name = name
        self._timepast = 0

    def __enter__(self):
        self._start = time.time()
        return self

    def __exit__(self, dtype, value, traceback):
        self._end = time.time()
        self._timepast = self._end - self._start
        print(u'{} : {}'.format(self._name, round(self._timepast, 5)))
