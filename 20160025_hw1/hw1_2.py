import sys
import time

THRESHOLD = 200
start_time = time.time()

with open(sys.argv[1], 'r') as f:
    lines = f.readlines()
    browsing = [[l for l in line.strip().split()] for line in lines]

browsing_items = {}
for line in browsing:
    for b in line:
        if b in browsing_items:
            browsing_items[b] += 1
        else:
            browsing_items[b] = 1
num_items = len(browsing_items)
freq_items = [b for b, i in browsing_items.items() if i > THRESHOLD]
freq_items_map = {b:i for i, b in enumerate(freq_items)}
num_freq_items = len(freq_items)

def triangle_index(i, j):
    i, j = (i, j) if i > j else (j, i)
    return int(i * (i-1) / 2) + j

def triangle_reverse(index):
    i = 1
    while index > i:
        index -= i
        i += 1
    return i, index

browsing_pairs = [0] * int(num_freq_items * (num_freq_items-1) / 2)
for line in browsing:
    for i, a in enumerate(line):
        for b in line[i+1:]:
            if a not in freq_items:
                continue
            if b not in freq_items:
                continue
            index = triangle_index(freq_items_map[a], freq_items_map[b])
            browsing_pairs[index] += 1

freq_pairs = [(triangle_reverse(i), n) for i, n in enumerate(browsing_pairs) if n > THRESHOLD]
num_freq_pairs = len(freq_pairs)
sorted_freq_pairs = sorted(freq_pairs, key=lambda x: x[1], reverse=True)

print(num_freq_items)
print(num_freq_pairs)
for (i, j), n in sorted_freq_pairs[:10]:
    print(freq_items[i], freq_items[j], n, sep='\t')

# print(f"elapsed time: {time.time() - start_time}")