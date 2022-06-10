import sys

if len(sys.argv) != 2:
    print("Usage: python3 replace_register_usage.py source.asm")
    exit(0)

f = open(sys.argv[1], "r")
lines = f.readlines()

# collect max private segment size
max_private_segment = 0
for line in lines:
    if line.find(".amdhsa_private_segment_fixed_size") != -1:
        parts = line.strip().split(" ")
        key = parts[0]
        value = int(parts[1])
        if value > max_private_segment:
            max_private_segment = value

need_private_segment = 0
if max_private_segment > 0:
    need_private_segment = 1


max_sgprs_per_simd = 800
max_vgprs_per_smid = 16 * 1024

max_sgprs_per_wave = 102
max_vgprs_per_wave = 256

sgpr_block = 8

sgprs_layers = [
    0,
    102,
    102,
    102,
    102,
    102,
    102,
    102,
    102,
    88,
    80
]

vgprs_layers = [
    0,
    256,
    128,
    84,
    64,
    48,
    40,
    36,
    32,
    28,
    28
]



def replace_text_segment_param(lines, key, value, values, key_word="_framework_"):
    new_lines = []

    current_kernel = ""

    for line in lines:
            # replace amdhas_next_free_vgpr
        if line.find(".amdhsa_kernel") != -1:
            kernel_name = line.strip().split(" ")[-1]
            current_kernel = kernel_name
            new_lines.append(line)
        elif line.find(key) != -1 and current_kernel.find(key_word) != -1 and current_kernel.split(key_word)[-1].isnumeric():
            num_layers = int(current_kernel.split(key_word)[-1])
            new_value= ""
            if value == None and values == None:
                # print("remove %s %s" % (current_kernel, key))
                continue
            if value != None:
                new_value = value
            else:
                new_value = values[num_layers]
            new_line = "		%s %d\n" % (key, new_value)
            # print("replace %s %s to %d" % (current_kernel, key, new_value))
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    return new_lines

def replace_symbol_segment_param(lines, key, value, values, key_word="_framework_"):
    new_lines = []

    current_kernel = ""

    for line in lines:
            # replace amdhas_next_free_vgpr
        if line.find(".name:") != -1:
            kernel_name = line.strip().split(" ")[-1]
            current_kernel = kernel_name
            new_lines.append(line)
        elif line.find(key) != -1 and current_kernel.find(key_word) != -1 and current_kernel.split(key_word)[-1].isnumeric():
            num_layers = int(current_kernel.split(key_word)[-1])
            new_value= ""
            if value != None:
                new_value = value
            else:
                new_value = values[num_layers]
            new_line = "    %s     %d\n" % (key, new_value)
            # print("replace %s %s to %d" % (current_kernel, key, new_value))
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    return new_lines    

def batch_replace(lines, key_word, private_segment):
    lines = replace_text_segment_param(lines, ".amdhsa_next_free_vgpr", None, vgprs_layers, key_word)
    lines = replace_text_segment_param(lines, ".amdhsa_next_free_sgpr", None, sgprs_layers, key_word)
    if private_segment:
        lines = replace_text_segment_param(lines, ".amdhsa_private_segment_fixed_size", max_private_segment, None, key_word)
        lines = replace_text_segment_param(lines, ".amdhsa_system_sgpr_private_segment_wavefront_offset", need_private_segment, None, key_word)
    # lines = replace_text_segment_param(lines, ".amdhsa_user_sgpr_flat_scratch_init", 0, None)
    lines = replace_text_segment_param(lines, ".amdhsa_user_sgpr_dispatch_ptr", 0, None, key_word)
    lines = replace_text_segment_param(lines, ".amdhsa_system_vgpr_workitem_id", 0, None, key_word)
    lines = replace_text_segment_param(lines, ".amdhsa_reserve_flat_scratch", None, None, key_word)
    lines = replace_text_segment_param(lines, ".amdhsa_reserve_vcc", None, None, key_word)

    lines = replace_symbol_segment_param(lines, ".vgpr_count:", None, vgprs_layers, key_word)
    lines = replace_symbol_segment_param(lines, ".sgpr_count:", None, sgprs_layers, key_word)
    if private_segment:
        lines = replace_symbol_segment_param(lines, ".private_segment_fixed_size:", max_private_segment, None, key_word)

    return lines

lines = batch_replace(lines, "merge_framework_", True)
lines = batch_replace(lines, "call_framework_", True)
lines = batch_replace(lines, "merge_framework_nostack_", False)
lines = batch_replace(lines, "proxy_kernel_", True)
lines = batch_replace(lines, "proxy_kernel_nostack_", False)
lines = replace_text_segment_param(lines, ".amdhsa_system_sgpr_workgroup_id_y", 1, None, "proxy_kernel_")
lines = replace_text_segment_param(lines, ".amdhsa_system_sgpr_workgroup_id_z", 1, None, "proxy_kernel_")
lines = replace_text_segment_param(lines, ".amdhsa_system_vgpr_workitem_id", 2, None, "proxy_kernel_")
lines = replace_text_segment_param(lines, ".amdhsa_system_sgpr_workgroup_id_y", 1, None, "proxy_kernel_nostack_")
lines = replace_text_segment_param(lines, ".amdhsa_system_sgpr_workgroup_id_z", 1, None, "proxy_kernel_nostack_")
lines = replace_text_segment_param(lines, ".amdhsa_system_vgpr_workitem_id", 2, None, "proxy_kernel_nostack_")
f.close()
f = open(sys.argv[1], "w")
f.writelines(lines)
f.close()
    # replace vgpr_count