import pandas as pd
import sys
import math

# Steps: 
# 1. profile the kernel: rocprof --hsa-trace ./raw_executor code_object schedule.json 1
# 2. delete the first warm up profile result.
# 3. use this file to estimate the potential throughput improvement.


if len(sys.argv) != 2:
    print("Usage: python3 estimate_max_throughput.py data.csv")
    exit(1)

num_cu = 80

df = pd.read_csv(sys.argv[1])
df = df[["grd", "wgr", "DurationNs"]]
df['blocks'] = df['grd'] // df['wgr']

df['cus'] = df['blocks'].map(lambda x: num_cu - math.ceil(x / math.ceil((x / num_cu))))
df['remain'] = df['cus'] * df['DurationNs']
total = df['DurationNs'].sum() * num_cu
remain = df['remain'].sum()

print("%f%%" % (remain / total * 100))