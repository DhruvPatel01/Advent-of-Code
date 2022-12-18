import re
from collections import Counter, defaultdict, deque
import itertools
from tqdm.autonotebook import tqdm
import numpy as np
import numba


# A lot of this code is inspired from Peter Norvig's style.

def id(x): return x

def mapt(fun, it): return tuple(map(fun, it))
def first(it, pred): return next(filter(pred, it))
def parse_uint(line): return mapt(int, re.findall(r'\d+', line))
def  parse_int(line): return mapt(int, re.findall(r'-?\d+', line))

def prod(it):
    r = 1
    for x in it: r *= x
    return r

        
def read(day, process=id, split='\n'):
    with open(f'./inputs/{day}.txt') as f:
        return mapt(process, f.read().rstrip('\n').split(split))
    
def partition_into(lst: list, when, preprocess=id, postprocess=id) -> list[list]:
    res = []
    for item in map(preprocess, lst):
        if when(item): 
            yield res
            res = []
        else: 
            res.append(postprocess(item))

def partition(inp, segment_length):
    for i in range(0, len(inp), segment_length):
        yield inp[i:i+segment_length]

def windows(inp, n):
    inp = tuple(inp)
    for i in range(len(inp)-n+1):
        yield inp[i:i+n]