import re
import sys
from pyspark import SparkConf, SparkContext

conf = SparkConf()
sc = SparkContext(conf=conf)

lines = sc.textFile(sys.argv[1])
friends = lines.map(lambda r: r.split('\t')) \
               .map(lambda r: (r[0], r[1].split(',')))

def combination(r):
    user, friends = r
    comb_list = [
        (a, b)
        for i, a in enumerate(friends)
        for b in friends[i+1:]
    ]
    comb_list += [
        (user, a) if user < a else (a, user)
        for a in friends
    ]
    return comb_list

common_friends = friends.flatMap(lambda r: [(pair, r[0]) for pair in combination(r)])
common_pair = common_friends.groupByKey().mapValues(list)
common_pair = common_pair.filter(lambda r: r[0][0] not in r[1])

num_common = common_pair.map(lambda r: (r[0], len(r[1])))
top_common = num_common.top(10, key=lambda r: r[1])
sorted(top_common, key=lambda r: r[1], reverse=True).foreach(
    lambda (a, b), n: print(a, b, n, sep='\t')
)

sc.stop()