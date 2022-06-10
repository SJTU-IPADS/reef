import pandas as pd
import sys
import math
import subprocess
import json


if len(sys.argv) < 2:
    print("Usage: python3 estimate_resource_usage.py model_dir")
    exit(1)

for i in range(1, len(sys.argv)):
    model_dir = sys.argv[i]
    if (model_dir.find("Makefile") != -1):
         continue
    _, model_name = subprocess.getstatusoutput("basename " + model_dir)

    model_profile = json.loads(open(model_dir + "/" + model_name + ".profile.json", "r").read())
    model_schedule = json.loads(open(model_dir + "/" + model_name + ".json", "r").read())

    num_cu = 60

    total = 0
    used = 0

    for kernel_info in model_schedule["kernels"]:
        blocks = kernel_info["launch_params"][0] * kernel_info["launch_params"][1] * kernel_info["launch_params"][2]
        cus = math.ceil(blocks / math.ceil((blocks / num_cu)))
        latency = model_profile[kernel_info["name"]]["total_latency"]
        total += latency * num_cu
        used += latency * cus

    print("%s: %f%%" % (model_name, used / total * 100))