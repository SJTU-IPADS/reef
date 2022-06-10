import sys
import json

############################################################
#
# This script is used to insert shared memory usage into the 
# model schedule json file.
#
# The shared memory usage is extracted from the device source
# code.
############################################################

if len(sys.argv) != 3:
    print("Usage: source_code.cpp schedule_file.json")
    exit(0)


def split_function_declaration(line):
    parts = line.split("(")
    parameters_str = parts[1].split(")")[0]
    left_parts = parts[0].split(" ")
    name = left_parts[-1]
    return_type = left_parts[-2]
    header = " ".join(left_parts[:-2])
    parameter_str_list = parameters_str.split(", ")
    parameters = []
    for param_str in parameter_str_list:
        parts = param_str.split(" ")
        param_name = parts[-1]
        param_type = " ".join(parts[:-1])
        parameters.append({"name": param_name, "type": param_type})
    return header, return_type, name, parameters



source_code_lines = open(sys.argv[1], "r").readlines()
schedule = json.loads(open(sys.argv[2], "r").read())

func_name = ""
shared_memory = 0

result = {}

for line in source_code_lines:
    if line.find("void") != -1:
        # save old values
        if func_name != "":
            if shared_memory < 4:
                shared_memory = 4
            result[func_name] = shared_memory

        _, _, curr_func_name, _ = split_function_declaration(line)
        func_name = curr_func_name
        shared_memory = 0
    if line.find("__shared__") != -1:
        # __shared__ float x[123];
        size = line.split("[")[1].split("]")[0]
        shared_memory = shared_memory + int(size) * 4

if func_name != "":
    if shared_memory < 4:
        shared_memory = 4
    result[func_name] = shared_memory

schedule["shared_memory"] = result

old_file_name = sys.argv[2].split(".json")[-2]
new_file_name = old_file_name + "_sm.json"
f = open(new_file_name, "w")
f.write(json.dumps(schedule))
f.close()