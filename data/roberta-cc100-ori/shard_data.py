import random
import os

f40 = [None for i in range(40)]
f40_v = [None for i in range(40)]
f40_t = [None for i in range(40)]


for i in range (40):
    f40[i] = open(f'shard/train.{i}.txt', 'w')  
    f40_v[i] = open(f'shard/valid.{i}.txt', 'w')
    f40_t[i] = open(f'shard/test.{i}.txt', 'w')


def random_fileno():
    return random.randint(0, 39)


fileno = 0
with open('split_data/train.txt') as f:
    line_cnt = 0
    while True:
        line = f.readline()
        if not line:
            break
        f40[fileno].write(line)
        line_cnt += 1
        if line_cnt % 1000 == 0:
            fileno = random_fileno()

for i in range (40):
     f40[i].close()

fileno = 0
with open('split_data/valid.txt') as f:
    line_cnt = 0
    while True:
        line = f.readline()
        if not line:
            break
        f40_v[fileno].write(line)
        line_cnt += 1
        if line_cnt % 1000 == 0:
            fileno = random_fileno()

for i in range (40):
    f40_v[i].close()


fileno = 0
with open('split_data/test.txt') as f:
    line_cnt = 0
    while True:
        line = f.readline()
        if not line:
            break
        f40_t[fileno].write(line)
        line_cnt += 1
        if line_cnt % 1000 == 0:
            fileno = random_fileno()

for i in range (40):
    f40_t[i].close()

