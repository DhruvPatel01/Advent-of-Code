#!/usr/bin/env python
# coding: utf-8

# In[92]:


from itertools import chain, permutations


# In[2]:


def read_snailno(s):
    depth = 0
    stack = []
    reg = 0
    prev_reg = True
    for c in s:
        if   c == '[':
            depth += 1
            prev_reg = False
        elif c == ']':
            if prev_reg:
                stack.append((reg, depth))
                reg = 0
            depth -= 1
            prev_reg = False
        elif c == ',':
            if prev_reg:
                stack.append((reg, depth))
                reg = 0
            prev_reg = False
        else:
            reg = reg*10 + int(c)
            prev_reg = True
    return stack


# In[41]:


def exploded_addtion(l, r):
    result = []
    carry = 0
    it = chain(l, r)
    for (reg, depth) in it:
        if depth >= 4:
            if result:
                (prev, pdepth) = result.pop()
                result.append((prev+reg+carry, pdepth))
            carry, _ = next(it)
            result.append((0, depth))
        else:
            result.append((reg+carry, depth+1))
            carry = 0
    return result


# In[42]:



# In[43]:


# import pprint
# pp = pprint.PrettyPrinter(indent=4, width = 10)
# pp.pprint(eval('[[[[[4,3],4],4],[7,[[8,4],9]]],[1,1]]'))


# In[44]:




# In[45]:


def split(s):
    result = []
    carry = 0
    it = iter(s)
    for reg, depth in it:
        reg += carry
        carry = 0
        
        if reg < 10: 
            result.append((reg, depth))
        else:
            a = reg//2
            b = reg - a
            if depth < 4:
                result.append((a, depth+1))
                result.append((b, depth+1))
                result.extend(it)
                return split(result)
            else:
                if result:
                    prev, pdepth = result.pop()
                    result.append((prev+a, pdepth))
                carry = b
                result.append((0, depth))
                for (reg, depth) in it:
                    result.append((reg+carry, depth))
                    carry = 0
                return split(result)
    return result


# In[46]:




# In[86]:


# inp = '''[[[0,[4,5]],[0,0]],[[[4,5],[2,6]],[9,5]]]
# [7,[[[3,7],[4,3]],[[6,3],[8,8]]]]
# [[2,[[0,8],[3,4]]],[[[6,7],1],[7,[1,6]]]]
# [[[[2,4],7],[6,[0,5]]],[[[6,8],[2,8]],[[2,1],[4,5]]]]
# [7,[5,[[3,8],[1,4]]]]
# [[2,[2,2]],[8,[8,1]]]
# [2,9]
# [1,[[[9,3],9],[[9,0],[0,7]]]]
# [[[5,[7,4]],7],1]
# [[[[4,2],2],6],[8,7]]'''

# inp = '''[[[0,[5,8]],[[1,7],[9,6]]],[[4,[1,2]],[[1,4],2]]]
# [[[5,[2,8]],4],[5,[[9,9],0]]]
# [6,[[[6,2],[5,6]],[[7,6],[4,7]]]]
# [[[6,[0,7]],[0,9]],[4,[9,[9,0]]]]
# [[[7,[6,4]],[3,[1,3]]],[[[5,5],1],9]]
# [[6,[[7,3],[3,2]]],[[[3,8],[5,7]],4]]
# [[[[5,4],[7,7]],8],[[8,3],8]]
# [[9,3],[[9,9],[6,[4,9]]]]
# [[2,[[7,7],7]],[[5,8],[[9,3],[0,2]]]]
# [[[[5,2],5],[8,[3,7]]],[[5,[7,5]],[4,4]]]'''

inp = open('../input/day18').read().strip()


# In[87]:


nos = list(map(read_snailno, inp.strip().split('\n')))


# In[88]:


l = nos[0]
for r in nos[1:]:
    l = exploded_addtion(l, r)
    l = split(l)


# In[89]:


def magnitude(s):
    stack = []
    for elem in s:
        stack.append(elem)
        while len(stack) > 1 and stack[-1][-1] == stack[-2][-1]:
            r, d = stack.pop()
            l, d = stack.pop()
            stack.append((3*l+2*r, d-1))
    return stack[0][0]


# In[91]:


print("Part1: ", magnitude(l))


# In[95]:


largest = float('-inf')
for l, r in permutations(nos, 2):
    n = magnitude(split(exploded_addtion(l, r)))
    if n > largest:
        largest = n


# In[97]:


print("Part2: ", largest)


# In[ ]:




