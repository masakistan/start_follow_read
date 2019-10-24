import sys

with open(sys.argv[1], 'r') as fh:
    for line in fh:
        f = line.strip().split()[0]
        f = f[f.rfind('_') + 1:f.rfind('.')]
        print(f)

