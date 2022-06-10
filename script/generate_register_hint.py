import sys
import json

def split_function_declaration(line):
    parts = line.strip().split("void")
    header = parts[0]
    return_type = "void"
    right_parts = parts[1].split("(")
    name = right_parts[0]
    parameters_str = right_parts[1].split(")")[0]
    parameter_str_list = parameters_str.split(", ")
    parameters = []
    for param_str in parameter_str_list:
        parts = param_str.split(" ")
        param_name = parts[-1]
        param_type = " ".join(parts[:-1])
        parameters.append({"name": param_name, "type": param_type})
    print(name)
    return header, return_type, name, parameters

def generate_function_declaration(return_type, name, params):
    params_str_list = []
    for param in params:
        param_str = param["type"] + " " + param["name"]
        params_str_list.append(param_str)
    return return_type + " " + name + "(" + ", ".join(params_str_list) + ")"


src = open(sys.argv[1], "r")
lines = src.readlines()

schedule = json.loads(open(sys.argv[2], "r").read())

new_lines = []

kernel_info = {}

for kernel in schedule["kernels"]:
    kernel_info[kernel["name"]] = kernel["launch_params"][3] * kernel["launch_params"][4] * kernel["launch_params"][5]

for line in lines:
    if line.find("__global__") != -1:
        _, _, func_name, params = split_function_declaration(line.strip())
        func_name = func_name.strip()
        if func_name in kernel_info:
            new_func = line.replace("void", "__attribute__((amdgpu_flat_work_group_size(%d, %d))) void" % (kernel_info[func_name], kernel_info[func_name]))
            new_lines.append(new_func)
            continue
    new_lines.append(line)

f = open(sys.argv[1], "w")
f.writelines(new_lines)
f.close()
