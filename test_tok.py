import json
import re
from collections import defaultdict
with open('data/seed/conversations.json', 'r') as f:
    data = json.load(f)
    corpus = ' '.join([d['input'] + ' ' + d['response'] for d in data.get('conversations', [])])
words = re.findall('\\w+|[^\\w\\s]', corpus, re.UNICODE)
splits = [[bytes([b]) for b in word.encode('utf-8')] for word in words[:1000]]
for i in range(100):
    counts = defaultdict(int)
    for split in splits:
        for j in range(len(split) - 1):
            counts[split[j], split[j + 1]] += 1
    if not counts:
        print(f'Stopped at {i} merges')
        break
    best_pair = max(counts.keys(), key=lambda k: counts[k])
    print(f'Merge {i}: {best_pair} count {counts[best_pair]}')
    new_splits = []
    for split in splits:
        j = 0
        new_split = []
        while j < len(split):
            if j < len(split) - 1 and split[j] == best_pair[0] and (split[j + 1] == best_pair[1]):
                new_split.append(best_pair[0] + best_pair[1])
                j += 2
            else:
                new_split.append(split[j])
                j += 1
        new_splits.append(new_split)
    splits = new_splits