import re
import sys
from pyspark import SparkConf, SparkContext

conf = SparkConf()
sc = SparkContext(conf=conf)

lines = sc.textFile(sys.argv[1])
words = lines.flatMap(lambda l: re.split(r'[^\w]+', l))

alphawords = words.filter(lambda w: len(w) > 0)
alphawords = alphawords.filter(lambda w: w[0].isalpha())

pairs = alphawords.map(lambda w: (w[0].lower(), 1))
counts = pairs.reduceByKey(lambda n1, n2: n1 + n2)

for c, n in counts.sortByKey().collect():
    print(f"{c}\t{n}")

sc.stop()