# this file contains the code that I actually wrote and submitted.
# The notebook code is the polished code, which has benefits of being
# written in hindsight.

## Day 7
input = read(7)[:-1]

tree = {'name': '/',
        'dirs': {},
        'files': {},
        'parent': None}
tree['parent'] = tree

line_no = 0
cwd = tree
while line_no < len(input):
    if input[line_no][2:] == 'ls':
        line_no += 1
        if line_no == len(input):
            break
        while input[line_no][0] != '$':
            a, b = input[line_no].split()
            if a == 'dir':
                if b not in cwd['dirs']:
                    cwd['dirs'][b] = {'name': b, 'dirs': {}, 'files': {}, 'parent': cwd}
            else:
                cwd['files'][b] = int(a)
            line_no += 1
            if line_no == len(input):
                break
    else:
        cmd, arg = input[line_no][2:].split()
        if arg == '/':
            cwd = tree
        elif arg == '..':
            cwd = cwd['parent']
        else:
            cwd = cwd['dirs'][arg]
        line_no += 1
        

def traverse(tree):
    s = 0
    for child in tree['dirs'].values():
        if 'size' in child:
            s += child['size']
        else:
            s += traverse(child)
    s += sum(tree['files'].values())
    tree['size'] = s      
    return s

traverse(tree)

def get_size(tree, th):
    s = 0
    if tree['size'] < th:
        s += tree['size']
    for child in tree['dirs'].values():
        s += get_size(child, th)
    return s

print(get_size(tree, 100000))

required = 30000000
available = 70000000 - tree['size']
need_to_free = required-available

def find_smallest(cwd, need):
    if cwd['size'] < need:
        return float('inf')
    sz = cwd['size']
    for child in cwd['dirs'].values():
        sz = min(sz, find_smallest(child, need))
    return sz

print(find_smallest(tree, need_to_free))

## Day 10

check_at = 20
cycle = 1
X = 1
part1 = 0
for inst in input:
    if cycle <= check_at <= cycle+1:
        part1 += X*check_at
        check_at += 40
    
    if inst == 'noop':
        cycle += 1
    else:
        cycle += 2
        X += int(inst.split()[1])
print(part1)


## Day 15 (Tool 9 minutes to run this code!)
can_not_exist = set()
for (x, y), r in sensor_rad.items():
    if target_row-r <= y <= target_row+r:
        width = r - abs(y-target_row)
        for dx in range(-width, width+1):
            can_not_exist.add(x+dx)
            
for (x, y) in beacon_pos:
    if y == target_row:
        if x in can_not_exist:
            can_not_exist.remove(x)

MX = 4000000+1
for target_row in tqdm(range(MX)):
    can_not_exist = set()
    for (x, y), r in sensor_rad.items():
        if target_row-r <= y <= target_row+r:
            width = r - abs(y-target_row)
            lb = max(0, x-width)
            ub = min(MX, x+width)
            can_not_exist.add((lb, ub))

        for (x, y) in beacon_pos:
            if 0 <= y <= MX:
                can_not_exist.add((y, y))

    length = 0
    it = iter(sorted(can_not_exist))
    l, u = next(it)
    if l > 0:
        print("This is the row", target_row)
        hi

    for lb, ub in it:
        if lb > u+1:
            print("This is the row", target_row)
            hi
            lbreak
        else:
            u = max(ub, u)
    if u < MX:
        print("This is the row", target_row)
        hi
