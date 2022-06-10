import sys
import json

f = open(sys.argv[1], "r")
lines = f.readlines()

descriptors = {}

is_descriptor = False

for line in lines:
    if line.find(".amdhsa_kernel ") != -1:
        is_descriptor = True
        continue
    if line.find(".end_amdhsa_kernel") != -1:
        is_descriptor = False
        continue
    if is_descriptor == False:
        continue
    parts = line.strip().split(" ")
    key = parts[0]
    value = parts[1]

    if key in descriptors:
        values = descriptors[key]
        values.append(value)
        descriptors[key] = list(set(values))
    else:
        values = []
        values.append(value)
        descriptors[key] = values


print(json.dumps(descriptors, sort_keys=True, indent=4))
    

        
    