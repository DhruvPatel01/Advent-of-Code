# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2


from helper import *

# + [markdown] tags=[]
# # Day 1
# -

input = partition_into(read(1), lambda x: not x, postprocess=int)
calories = [sum(l) for  l in input]
calories.sort()
assert calories[-1] == 66487
assert sum(calories[-3:]) == 197301

# + [markdown] tags=[]
# # Day 2
# -

scores = {
    'A X': 1+3,
    'A Y': 2+6,
    'A Z': 3+0,
    'B X': 1+0,
    'B Y': 2+3,
    'B Z': 3+6,
    'C X': 1+6,
    'C Y': 2+0,
    'C Z': 3+3,
}

input = read(2)

assert sum(map(scores.get, input)) == 10994

scores2 = {
    'A X': 3+0,
    'A Y': 1+3,
    'A Z': 2+6,
    'B X': 1+0,
    'B Y': 2+3,
    'B Z': 3+6,
    'C X': 2+0,
    'C Y': 3+3,
    'C Z': 1+6,
}

assert sum(map(scores2.get, input)) == 12526

# # Day 3

# +
import string

priority = dict(enumerate(string.ascii_lowercase, 1))
priority |= dict(enumerate(string.ascii_uppercase, 27))
priority = {v:k for k,v in priority.items()}
# -

input = read(3, str.strip)


# +
def split_half (x):
    n = len(x)//2
    return set(x[:n]), set(x[n:])

common = [x&y for x, y in map(split_half, input)]
# -

asset = sum(sum(map(priority.get, c)) for c in common) == 7848

groups = [input[i:i+3] for i in range(0, len(input), 3)]
groups = [list(set(x)&set(y)&set(z))[0] for x, y, z in groups]
assert sum(map(priority.get, groups)) == 2616

# # Day 4

input = read(4, parse_uint)


def overlap(args):
    a, b, c, d = args
    return a <= c <= d <= b or c <= a <= b <= d


assert sum(map(overlap, input)) == 582


def partial(args):
    a, b, c, d = args
    return a <= c <= b <= d or c <= a <= d <= b or overlap(args)


assert sum(map(partial, input)) == 893

# # Day 5

input = read(5, lambda x: f' {x}')

# +
crates = [[] for _ in range(10)]
for i in range(8):
    for j, elem in enumerate(partition(input[i], 4), 1):
        if elem.strip():
            crates[j].append(elem[2])

crates_orig = [crate[::-1] for crate in crates]
# -

crates = [crate.copy() for crate in crates_orig]
for line in input[10:]:
    a, b, c = parse_int(line)
    for j in range(a):
        crates[c].append(crates[b].pop())

assert ''.join(crate[-1] for crate in crates[1:]) == 'CWMTGHBDW'

crates = [crate.copy() for crate in crates_orig]
for line in input[10:]:
    a, b, c = parse_int(line)
    crates[c].extend(crates[b][-a:])
    crates[b] = crates[b][:-a]

assert ''.join(crate[-1] for crate in crates[1:]) == 'SSCGWJCRB'

# ## Day 6

input = read(6)[0]


def solve_6(input, L):
    for i, window in enumerate(windows(input, L)):
        if len(set(window)) == L:
            return i+L


assert solve_6(input, 4) == 1480
assert solve_6(input, 14) == 2746

# # Day 7

# +
# Note: This is not the first version of the code I wrote.
# The first version was messy, and is located inside `original.py`.
# It simulated the whole filesystem as a tree, and did a depth-first-search.
# -

input = read(7, split='$')[1:]

cwd = []
dirs = Counter()
for command in input:
    command, *output = command.strip().split()
    match command:
        case 'cd':
            match output[0]:
                case '/': 
                    cwd = ['/']
                case '..': 
                    if cwd: cwd.pop()
                case d:
                    cwd.append(d)
        case 'ls':
            if '/'.join(cwd) not in dirs:
                s = sum(int(a) for a, b in partition(output, 2) if a != 'dir')
                for i in range(len(cwd)):
                    dirs['/'.join(cwd[:i+1])] += s


assert sum(v for v in dirs.values() if v <= 100000) == 1306611

need = 30000000 - (70000000-dirs['/'])
assert min(v for v in dirs.values() if v >= need) == 13210366

# # Day 8

import numpy as np

input = read(8)
# input = '''30373
# 25512
# 65332
# 33549
# 35390'''.split()


input = [mapt(int, line) for line in input]

grid = np.array(input)
h, w = grid.shape

td = np.maximum.accumulate(grid, axis=0)
lr = np.maximum.accumulate(grid, axis=1)
dt = np.maximum.accumulate(grid[::-1, :], axis=0)[::-1, :]
rl = np.maximum.accumulate(grid[:, ::-1], axis=1)[:, ::-1]

td = td[0:-2, 1:-1]
lr = lr[1:-1, 0:-2]
dt = dt[2:, 1:-1]
rl = rl[1:-1, 2:]

grid = grid[1:-1, 1:-1]

flag = grid > td
flag |= grid > dt
flag |= grid > lr
flag |= grid > rl

flag.sum() + h*2 + (w-2)*2

grid = np.array(input)
h, w = grid.shape

mx = 0
for i in range(h):
    for j in range(w):
        col = (grid[:, j] - grid[i, j]) >= 0
        
        try:
            t = col[:i][::-1].tolist().index(True)+1
        except: 
            t = i
            
        try:
            b = col[i+1:].tolist().index(True)+1
        except:
            b = h-i-1
        
        
        row = (grid[i, :] - grid[i, j]) >= 0
        try:
            l = row[:j][::-1].tolist().index(True)+1
        except:
            l = j
            
        try:
            r = row[j+1:].tolist().index(True)+1
        except:
            r = w-j-1
        
        mx = max(mx, t*b*l*r)

mx

w

# # Day 9

# +
input = '''R 4
U 4
L 3
D 1
R 4
D 1
L 5
R 2'''.split()

input = '''R 5
U 8
L 8
D 3
R 17
D 10
L 25
U 20'''.split()
# -

input = read(9, split=None)

# +
hr = hc = tr = tc = 0

visited = {(0, 0)}
for command in partition(input, 2):
    match command:
        case ('U', x):
            for _ in range(int(x)):
                hr += 1
                if tr < hr-1:
                    tc = hc
                    tr = hr-1
                    visited.add((tr, tc))
                # print(command, tr, tc)
        case ('D', x):
            for _ in range(int(x)):
                hr -= 1
                if tr > hr+1:
                    tc = hc
                    tr = hr+1
                    visited.add((tr, tc))
                # print(command, tr, tc)
        case ('L', x):
            for _ in range(int(x)):
                hc -= 1
                if tc > hc+1:
                    tr = hr
                    tc = hc+1
                    visited.add((tr, tc))
                # print(command, tr, tc)
        case ('R', x):
            for _ in range(int(x)):
                hc += 1
                if tc < hc-1:
                    tr = hr
                    tc = hc-1
                    visited.add((tr, tc))
                # print(command, tr, tc)

# -

def vis_grid(pos):
    min_r = min(p[0] for p in pos)
    min_c = min(p[1] for p in pos)
    
    width = 50
    height = 50
    
    grid = [['.' for _ in range(width)] 
            for _ in range(height)]
    for i, (r, c) in enumerate(pos[::-1]):
        grid[min_r+r][min_c+c] = chr(ord('9') - i)
    
    return '\n'.join(''.join(line) for line in grid[::-1])


# + tags=[]
# pos = [[0,0] for i in range(10)]

# visited = {(0, 0)}
# for (command, repeat) in partition(input, 2):
#     print('X'*40)
#     print(vis_grid(pos))
#     # print(pos)
#     for i in range(int(repeat)):
#         hr, hc = pos[0]
#         for j in range(1, 10):
#             tr, tc = pos[j]
#             moved = False
#             match command:
#                 case 'U':
#                     hr += 1
#                     if tr < hr-1:
#                         tc = hc
#                         tr = hr-1
#                         moved = True
#                 case 'D':
#                     hr -= 1
#                     if tr > hr+1:
#                         tc = hc
#                         tr = hr+1
#                         moved = True
#                 case 'L':
#                     hc -= 1
#                     if tc > hc+1:
#                         tr = hr
#                         tc = hc+1
#                         moved = True
#                 case 'R':
#                     hc += 1
#                     if tc < hc-1:
#                         tr = hr
#                         tc = hc-1
#                         moved = True
#             pos[j-1] = hr, hc
#             hr, hc = tr, tc
#             if not moved: break
#         visited.add((tr, tc))
#         pos[9] = (tr, tc)

# +

# min_r = min(p[0] for p in pos)
# min_c = min(p[1] for p in pos)
# width = 30
# height = 30

# grid = [['.' for _ in range(width)] 
#         for _ in range(height)]
# for i, (r, c) in enumerate(pos[::-1]):
#     grid[min_r+r][min_c+c] = chr(ord('9') - i)

# + tags=[]
# print(vis_grid(pos))
# -

# # Day 10

input = read(10, split='\n')

Xs = [1]
for inst in input:
    Xs.append(Xs[-1])
    if inst != 'noop': Xs.append(Xs[-1]+int(inst.split()[-1]))


sum(i*v for i, v in enumerate(Xs, 1) if (i - 20)%40 == 0)

# +
C = 0
X = 1
output = []
row = []

def draw():
    global row        
    c = C%40 
    row.append('█'  if X-1 <= c <= X+1 else ' ')
    if len(row) == 40:
        output.append(row)
        row = []
   
for inst in input:
    draw()
    if inst == 'noop':
        C += 1
    else:
        C += 1
        draw()
        C += 1
        X += int(inst.split()[1])
# -

print('\n'.join(''.join(line) for line in output))


# # Day 11

def parse_monkey(s):
    s = s.split('\n')
    items = list(parse_uint(s[1]))
    op = s[2].split('=')[1]
    test = parse_uint(s[3])[0]
    true = parse_uint(s[4])[0]
    false = parse_uint(s[5])[0]
    
    return (items, op, test, true, false)


def round(inspects):
    for i, d in enumerate(data):
        inspects[i] += len(d[0])
        for old in d[0]:
            new = eval(d[1])
            new //= 3
            if new % d[2] == 0:
                data[d[3]][0].append(new)
            else:
                data[d[4]][0].append(new)
        d[0].clear()


input = read(11, split='\n\n')

# +
data = [parse_monkey(s) for s in input]
inspects = [0]*len(data)
for r in range(20):
    round(inspects)

assert prod(sorted(inspects)[-2:]) == 120056


# + tags=[]
def round2(inspects):
    for i, d in enumerate(data):
        inspects[i] += len(d[0])
        for old in d[0]:
            new = eval(d[1])
            if new % d[2] == 0:
                data[d[3]][0].append(new % div)
            else:
                data[d[4]][0].append(new % div)
        d[0].clear()


# +
data = [parse_monkey(s) for s in input]
div = prod(d[2] for d in data)
inspects = [0]*len(data)
for r in tqdm(range(10000)):
    round2(inspects)

assert prod(sorted(inspects)[-2:]) == 21816744824

# -

# ## Day 12

# +
input = read(12)

grid = [[ord('z')+2]*(len(input[0])+2)]
for i, line in enumerate(input, 1):
    row = [ord('z')+2]
    row.extend(map(ord, line))
    row.append(ord('z')+2)
    if 'S' in line:
        start = (i, line.index('S')+1)
        row[start[1]] = ord('a')
    if 'E' in line:
        stop = (i, line.index('E')+1)
        row[stop[1]] = ord('z')
    grid.append(row)
grid.append(grid[0])

h, w = len(grid), len(grid[0])

def neighbors(i, j):
    if grid[i][j]+1 >= grid[i-1][j]: yield i-1, j
    if grid[i][j]+1 >= grid[i][j+1]: yield i, j+1
    if grid[i][j]+1 >= grid[i+1][j]: yield i+1, j
    if grid[i][j]+1 >= grid[i][j-1]: yield i, j-1
        
def bfs(start):
    steps = {k: 0 for k in start}
    heap = deque(start)
    
    while heap:
        pos = heap.popleft()
        if pos == stop:  return steps[pos]
        
        d = steps[pos]
        for n in neighbors(*pos):
            if n not in steps:
                steps[n] = d + 1
                heap.append(n)



# -

assert bfs([start])  == 437

starts = [(i, j) for i, row in enumerate(grid)
              for j, c in enumerate(row) if c == ord('a')]


assert bfs(starts) == 430


# # Day 13

def compare(a, b):
    match a, b:
        case list(), list():
            for aa, bb in zip(a, b):
                r = compare(aa, bb)
                if r != 0: return r
            return len(a)-len(b)
        case int(), list():
            return compare([a], b)
        case list(), int():
            return compare(a, [b])
        case _:
            return a-b


input = read(13, split='\n\n',process=lambda x: mapt(eval, x.split('\n')))

assert sum(i*(compare(a, b) < 0) for i, (a, b) in enumerate(input, 1)) == 5825

# +
from functools import cmp_to_key

input = list(itertools.chain(*input))
input.extend(([[2]], [[6]]))
input.sort(key=cmp_to_key(compare))

assert (input.index([[2]])+1)*(input.index([[6]])+1) == 24477
# -

# # Day 14

input = read('14', process=parse_uint)

# +
blocked = set()

def block_hline(y, x1, x2):
    if x1 > x2: x1, x2 = x2, x1
    for x in range(x1, x2+1): blocked.add((x, y))
    
def block_vline(x, y1, y2):
    if y1 > y2: y1, y2 = y2, y1
    for y in range(y1, y2+1): blocked.add((x, y))                                      

def init_day14():
    blocked.clear()
    for line in input:
        for (xx, yy), (x, y) in windows(partition(line, 2), 2):
            if x == xx: block_vline(x, yy, y)
            else      : block_hline(y, xx, x)
        
def simulate(abyss, is_unblocked):
    count = 0
    x, y = (500, 0)
    while True:
        if abyss(x, y):
            break
        elif is_unblocked(x, y+1):
            y = y+1
        elif is_unblocked(x-1, y+1):
            x, y = x-1, y+1
        elif is_unblocked(x+1, y+1):
            x, y = x+1, y+1
        else:
            blocked.add((x, y))
            count += 1
            x, y = 500, 0
           
            if (x, y) in blocked:
                break
    return count


# -

init_day14()
Y = max(p[1] for p in blocked)
assert simulate(lambda x, y: y > Y, lambda x, y: (x, y) not in blocked) == 779

init_day14()
Y = max(p[1] for p in blocked)+2
assert simulate(abyss=lambda x, y: abs(x-500) > y, 
                is_unblocked=lambda x, y: not(y == Y or (x, y) in blocked)) == 27426

# # Day 15

input = read('15', parse_int)


@numba.njit
def find_interesting_locations(input, target_row, MN, MX):
    segments = []
    for x, y, x1, y1 in input:
        r = abs(x-x1)+abs(y-y1)
        if target_row-r <= y <= target_row+r:
            width = r - abs(y-target_row)
            segments.append((max(MN, x-width), min(MX, x+width)))
        
        if MN <= y1 < MX:
            segments.append((y1, y1))
    
    segments.sort()
    answer = 0
    L, U = segments[0]
    breakpt = None
    for l, u in segments[1:]:
        if l <= U+1:
            U = max(u, U)
        else:
            breakpt = U+1
            answer += (U-L)
            L, U = l, u
    answer += (U-L)
    return answer, breakpt


MN = float('-inf')
MX = float('inf')
assert int(find_interesting_locations(input, 2000000, MN, MX)[0]) == 5256611


@numba.njit
def find_discontinuity(input):
    for row in range(0, 4000001):
        _, dis = find_interesting_locations(input, row, 0, 4000_000)
        if dis is not None:
            return dis*4000000+row


assert find_discontinuity(input) == 13337919186981

# # Day 16

import itertools


# +
def parse_nodes(line):
    return re.findall(r'[A-Z]{2}', line)

input = read('16', parse_nodes)

graph = defaultdict(set)
for u, *vs in input:
    graph[u].update(vs)

flow_rates = {i[0]: j[0] for i, j in zip(input, read('16', parse_uint))}

# +
valid = set(sorted(k for k, v in flow_rates.items() if v > 0))
# valid = {'HH', 'DD'}
subsets = set()
for i in range(1, len(valid)+1):
    subsets.update(itertools.combinations(valid, i))

nextlevel = defaultdict(int)
for subset in subsets:
    for s in subset:
        nextlevel[s, subset] = flow_rates[s]
    
for level in tqdm(range(28, -1, -1)):
    current = defaultdict(int)
    for subset in subsets:
        for s in subset:
            val1 = nextlevel[s, tuple(t for t in subset if t!=s)]
            val1 += (30-level-1)*flow_rates[s]
            
            val2 = max(nextlevel[t, subset] for t in graph[s])
            current[s, subset] = max(val1, val2)
        for s in set(graph) - set(subset):
            current[s, subset] = max(nextlevel[t, subset] for t in graph[s])
    nextlevel = current
    del current

# -

max(v for k, v in nextlevel.items() if k[0] == 'AA')



# +
valid = set(sorted(k for k, v in flow_rates.items() if v > 0))
idx = {k:i for i, k in enumerate(valid)}
for k in set(graph)-valid:
    idx[k] = len(idx)
valid = sorted([idx[v] for v in valid])
all_nodes = np.arange(len(graph)).astype(int)

graph_new = [[255]*5 for i in range(len(graph))]
for node in graph:
    u = idx[node]
    for i, nb in enumerate(graph[node]):
        graph_new[u][i] = idx[nb]

graph_new = np.array(graph_new)


# +
subsets = set()
for i in range(1, len(valid)+1):
    subsets.update(itertools.combinations(valid, i))
    
valid = set(valid)
# -

flow_rates_np = np.zeros_like(all_nodes)
for node, fl in flow_rates.items():
    flow_rates_np[idx[node]] = fl


# +
def get_subset_idx(subset):
    r = 0
    for i in subset:
        r |= 1 << i
    return r
        
def get_idx_into(s, t, subset):
    r = get_subset_idx(subset)
    r |= s << 15
    r |= t << (15+6)
    return r


# +
@numba.njit
def fill_good_good_pos(level, subset_idx, subset, graph, flow_rate, nextlevel, current,):
    
    for s in subset:
        for t in subset:
            if s >= t: continue
            
            new = 0
            for u in subset:
                if u != s and u != t:
                    new |= 1 << u
                        
            opt1 = (26-level-1)*(flow_rates[s]+flow_rates[t])
            opt1 += nextlevel[new | (s << 15) | (t << 21)]
            
            new = 0
            for u in subset:
                if u != s: new |= 1 << u
            opt2 = (26-level-1)*(flow_rates[s])
            mx = 0
            for tt in graph[t]:
                if tt == 255: continue
                mx = max(nextlevel[new | (s << 15) | (tt << 21)], mx)
            opt2 += mx
            
            new = 0
            for u in subset:
                if u != t: new |= 1 << u
            
            opt3 = (26-level-1)*(flow_rates[t])
            for ss in graph[s]:
                if ss == 255: continue
                mx = max(nextlevel[new | (ss << 15) | (t << 21)], mx)
                
            mx = 0
            for ss in graph[s]:
                if ss==255: break
                for tt in graph[t]:
                    if tt == 255: break
                    mx = max(mx, nextlevel[subset_idx | (ss << 15) | (tt << 21)])
            opt4 = mx
            
            current[subset_idx | (s << 15) | (t << 21)] = max(opt1, opt2, opt3, opt4)
            current[subset_idx | (t << 15) | (s << 21)] = max(opt1, opt2, opt3, opt4)

@numba.njit
def fill_bad_bad_pos(subset_idx, graph, others, nextlevel, current):
    for s in others:
        for t in others:
            mx = 0
            for ss in graph[s]:
                if ss == 255: break
                for tt in graph[t]:
                    if tt == 255: break
                    mx = max(mx, nextlevel[subset_idx | (ss<<15) | (tt<<21)])          
            
            current[subset_idx | (s << 15) |  (t << 21)] = mx
            
@numba.njit
def fill_good_bad_pos(level, flow_rates_np, graph, subset_idx, subset, others, nextlevel, current):
    for s in subset:
        new = 0
        for t in subset:
            if t != s: new |= 1 << t
        opt = (26-level-1)*(flow_rates_np[s])
        
        for t in others:
            mx = 0
            for tt in graph[t]:
                if tt == 255: break
                mx = max(nextlevel[new | (s << 15) | (tt << 21)], mx)
            opt1 = opt + mx

            mx = 0
            for ss in graph[s]:
                if ss == 255: break
                for tt in graph[t]:
                    if tt == 255: break
                    mx = max(mx, nextlevel[subset_idx | (ss << 15) | (tt << 21)])
                    
            current[subset_idx | (s << 15) | (t << 21)] = max(opt1, mx)
            current[subset_idx | (t << 15) | (s << 21)] = max(opt1, mx)


# +
nextlevel = np.zeros(2**(6+6+15), dtype=np.uint32)
for subset in tqdm(subsets):
    subset_idx = get_subset_idx(subset)
    for s, t in itertools.combinations(subset, 2):
        nextlevel[subset_idx | (s << 15) | (t << 21)] = flow_rates_np[s]+flow_rates_np[t]
        nextlevel[subset_idx | (t << 15) | (s << 21)] = flow_rates_np[s]+flow_rates_np[t]
                            
        
    for s in subset:
        nextlevel[subset_idx | (s << 15) | (s << 21)] = flow_rates_np[s]
        
        for t in set(all_nodes)-set(subset):
            nextlevel[subset_idx | (s << 15) | (t << 21)] = flow_rates_np[s]
            nextlevel[subset_idx | (t << 15) | (s << 21)] = flow_rates_np[s]

            
for level in tqdm(range(24, -1, -1)):
    current = np.zeros(2**(6+6+15), dtype=np.uint32)
    for subset in tqdm(subsets):
        others = np.array(list(set(all_nodes) - set(subset)))
        subset_idx = get_subset_idx(subset)
        for s, t in itertools.combinations(subset, 2):
            new = get_subset_idx(u for u in subset if u != s and u != t)
            opt1 = (26-level-1)*(flow_rates_np[s]+flow_rates_np[t])
            opt1 += nextlevel[new | (s << 15) | (t << 21)]
            
            new = get_subset_idx(u for u in subset if u != s)
            opt2 = (26-level-1)*(flow_rates_np[s])
            opt2 += max(nextlevel[new | (s << 15) | (tt << 21)] for tt in graph_new[t] if tt < 255)
            
            new = get_subset_idx(u for u in subset if u != t)
            opt3 = (26-level-1)*(flow_rates_np[t])
            opt3 += max(nextlevel[new | (ss << 15) | (t << 21)] for ss in graph_new[s] if ss < 255)
            
            opt4 = max(nextlevel[subset_idx | (ss << 15) | (tt << 21)]
                       for ss in graph_new[s] for tt in graph_new[t] if tt < 255 and ss < 255)
            
            current[subset_idx | (s << 15) | (t << 21)] = max(opt1, opt2, opt3, opt4)
            current[subset_idx | (t << 15) | (s << 21)] = max(opt1, opt2, opt3, opt4)
            
        for s in subset:
            new = get_subset_idx(u for u in subset if u != s)
            opt1 = (26-level-1)*(flow_rates_np[s])
            opt1 += max(nextlevel[new | (s << 15) | (tt << 21)] for tt in graph_new[s] if tt < 255)
            opt2 = max(nextlevel[subset_idx | (ss << 15) | (tt << 21)]
                       for ss in graph_new[s] for tt in graph_new[s] if tt < 255 and ss < 255)             
            current[subset_idx | (s << 15) | (s << 21)] = max(opt1, opt2)
            
        fill_good_bad_pos(level, flow_rates_np, graph_new, subset_idx, np.array(subset), others, nextlevel, current)
        fill_bad_bad_pos(subset_idx, graph_new, others, nextlevel, current)
        
        
    nextlevel = current
    del current

# -

max(nextlevel[get_idx_into(idx['AA'], idx['AA'], subset)] for subset in subsets)

# That was wrong. Sigh:(

# # Day 18

import sys
sys.setrecursionlimit(10000)

input = sorted(read('18', parse_uint))


def build_graph(input):
    graph = {k: set() for k in sorted(input)}
    
    for node in graph:
        x,y,z = node
        adj = {(x-1, y, z), (x+1, y, z),
               (x, y-1, z), (x, y+1, z),
               (x, y, z-1), (x, y, z+1)}
        adj = {u for u in adj if u in graph}
        graph[node].update(adj)
    
    return graph


def run_dfs(graph, visited, node) -> int:
    visited.add(node)
    res = 6 - len(graph[node])
    for v in graph[node]:
        if v not in visited:
            res += run_dfs(graph, visited, v)
    return res


graph = build_graph(input)

part1 = 0
visited = set()
for node in graph:
    if node not in visited:
        part1 += run_dfs(graph, visited, node)


assert part1 == 4340

min_x = min(node[0] for node in graph)-1
min_y = min(node[1] for node in graph)-1
min_z = min(node[2] for node in graph)-1
max_x = max(node[0] for node in graph)+1
max_y = max(node[1] for node in graph)+1
max_z = max(node[2] for node in graph)+1


# +
eps = set()
for y in range(min_y, max_y+1):
    for z in range(min_z, max_z+1):
        pt1 = (min_x, y, z)
        pt2 = (max_x, y, z)
        eps.add(pt1)
        eps.add(pt2)
        
for x in range(min_x, max_x+1):
    for z in range(min_z, max_z+1):
        pt1 = (x, min_y, z)
        pt2 = (x, max_y, z)
        eps.add(pt1)
        eps.add(pt2)
        
for x in range(min_x, max_x+1):
    for y in range(min_y, max_y+1):
        pt1 = (x, y, min_z)
        pt2 = (x, y, max_z)
        eps.add(pt1)
        eps.add(pt2)


# -

def run_dfs2(graph, visited, node) -> int:
    x, y, z = node
    if x < min_x or x > max_x or y < min_y or y > max_y or z < min_z or z > max_z:
        return 0
    visited.add(node)
    adj = {(x-1, y, z), (x+1, y, z),
           (x, y-1, z), (x, y+1, z),
           (x, y, z-1), (x, y, z+1)}
    
    res = 0
    for u in adj:
        if u in graph: res += 1
        elif u not in visited:
            res += run_dfs2(graph, visited, u)
    return res


part2 = 0
visited = set()
for ep in eps:
    if ep not in visited:
        part2 += run_dfs2(graph, visited, ep)

assert part2 == 2468

# # Day 19


