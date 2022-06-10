import sys


if len(sys.argv) != 3:
    print("Usage: python3 replace_raw_occupancy.py raw_source.cu trans_asm.s")

def get_occupancy(lines):
    current_kernel = ""
    occupancy = {}
    stack_size = {}
    for line in lines:
        if line.find(".amdhsa_kernel") != -1:
            kernel_name = line.strip().split(" ")[-1]
            current_kernel = kernel_name
        if line.find("Occupancy") != -1:
            occupancy[current_kernel] = int(line.strip().split(" ")[-1])
        if line.find("ScratchSize") != -1:
            stack_size[current_kernel] = int(line.strip().split(" ")[-1])

    return occupancy, stack_size

asm = open(sys.argv[2], "r")
asm_lines = asm.readlines()
asm.close()

asm_occupancy, _ = get_occupancy(asm_lines)

raw_source = open(sys.argv[1], "r")
raw_lines = raw_source.readlines()
raw_source.close()

new_lines = []

for line in raw_lines:
    if line.find('__global__') != -1:
        parts = line.split("void")
        right_part = parts[1]
        func_name = right_part.split("(")[0].strip()
        left_part = 'extern "C" __global__ __attribute__((amdgpu_waves_per_eu(%d,%d))) void ' % (asm_occupancy[func_name], asm_occupancy[func_name])
        new_line = left_part + right_part
        new_lines.append(new_line)
        continue
    new_lines.append(line)

raw_source = open(sys.argv[1], "w")
raw_source.writelines(new_lines)
raw_source.close()

