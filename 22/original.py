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