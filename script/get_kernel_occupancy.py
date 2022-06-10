import sys

if len(sys.argv) != 3:
    print("Usage: python3 get_kernel_occupancy.py raw.s trans.s")
    exit(1)

raw_asm = open(sys.argv[1], "r")
trans_asm = open(sys.argv[2], "r")

raw_lines = raw_asm.readlines()
trans_lines = trans_asm.readlines()

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


raw_occupancy, raw_stack = get_occupancy(raw_lines)
trans_occupancy, trans_stack = get_occupancy(trans_lines)

print("Occupancy:")
for kernel, occupancy in raw_occupancy.items():
    if occupancy > trans_occupancy[kernel]:
        print("%s: %d, %d" %(kernel, occupancy, trans_occupancy[kernel]))

print("Stack:")
for kernel, stack in raw_stack.items():
    if stack < trans_stack[kernel]:
        print("%s: %d, %d" %(kernel, stack, trans_stack[kernel]))