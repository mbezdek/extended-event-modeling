import sys

assert len(sys.argv) == 4, "Must have two input lists and one output list"

first = open(sys.argv[1], 'r').readlines()
first = [x.strip() for x in first]
second = open(sys.argv[2], 'r').readlines()
second = [x.strip() for x in second]
diff = list(set(first).difference(set(second)))
open(sys.argv[3], 'w').writelines('\n'.join(diff))
