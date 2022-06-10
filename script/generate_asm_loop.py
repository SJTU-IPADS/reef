import os
import sys
import json

##
## This script is used to generate assembly codes that implements persistent thread loops.
##
## Usage: python3 generate_asm_loop.py kernel.s 
##
##

def generate_device_func_header():
    template = """
    .text
    .amdgcn_target "amdgcn-amd-amdhsa--gfx906"
    .p2align    2                               ; -- Begin function get_3d_idx
    .type    get_3d_idx,@function
get_3d_idx:                   ; @get_3d_idx
; %bb.0:

;input idx(v0) dim(s0, s1)
;output idx(v1, v2, v3)
;jump_back: s10, s11
;usage: v0 - v6, s0 - s8
    s_mov_b32 s8, 0x4f7ffffe
    s_waitcnt lgkmcnt(0)
    v_cvt_f32_u32_e32 v1, s0
    v_cvt_f32_u32_e32 v2, s1
    s_sub_i32 s4, 0, s0
    s_mul_i32 s6, s1, s0
    v_rcp_iflag_f32_e32 v1, v1
    v_rcp_iflag_f32_e32 v2, v2
    v_cvt_f32_u32_e32 v3, s6
    s_sub_i32 s5, 0, s1
    v_mul_f32_e32 v1, s8, v1
    v_cvt_u32_f32_e32 v1, v1
    v_mul_f32_e32 v2, s8, v2
    v_cvt_u32_f32_e32 v2, v2
    v_rcp_iflag_f32_e32 v3, v3
    v_mul_lo_u32 v4, s4, v1
    s_sub_i32 s7, 0, s6
    v_mul_lo_u32 v5, s5, v2
    v_mul_f32_e32 v3, s8, v3
    v_mul_hi_u32 v4, v1, v4
    v_cvt_u32_f32_e32 v3, v3
    v_mul_hi_u32 v5, v2, v5
    v_add_u32_e32 v1, v1, v4
    v_mul_hi_u32 v1, v0, v1
    v_mul_lo_u32 v4, s7, v3
    v_add_u32_e32 v2, v2, v5
    v_mul_lo_u32 v6, v1, s0
    v_add_u32_e32 v5, 1, v1
    v_mul_hi_u32 v4, v3, v4
    v_sub_u32_e32 v6, v0, v6
    v_cmp_le_u32_e32 vcc, s0, v6
    v_cndmask_b32_e32 v1, v1, v5, vcc
    v_subrev_u32_e32 v5, s0, v6
    v_cndmask_b32_e32 v5, v6, v5, vcc
    v_add_u32_e32 v6, 1, v1
    v_cmp_le_u32_e32 vcc, s0, v5
    v_cndmask_b32_e32 v5, v1, v6, vcc
    v_mul_hi_u32 v1, v5, v2
    v_add_u32_e32 v2, v3, v4
    v_mul_hi_u32 v3, v0, v2
    v_mul_lo_u32 v2, v5, s0
    v_mul_lo_u32 v4, v1, s1
    v_mul_lo_u32 v6, v3, s6
    v_sub_u32_e32 v1, v0, v2
    v_sub_u32_e32 v2, v5, v4
    v_subrev_u32_e32 v4, s1, v2
    v_cmp_le_u32_e32 vcc, s1, v2
    v_cndmask_b32_e32 v2, v2, v4, vcc
    v_subrev_u32_e32 v4, s1, v2
    v_cmp_le_u32_e32 vcc, s1, v2
    v_cndmask_b32_e32 v2, v2, v4, vcc
    v_sub_u32_e32 v4, v0, v6
    v_add_u32_e32 v5, 1, v3
    v_cmp_le_u32_e32 vcc, s6, v4
    v_cndmask_b32_e32 v3, v3, v5, vcc
    v_subrev_u32_e32 v5, s6, v4
    v_cndmask_b32_e32 v4, v4, v5, vcc
    v_add_u32_e32 v5, 1, v3
    v_cmp_le_u32_e32 vcc, s6, v4
    v_cndmask_b32_e32 v3, v3, v5, vcc
    v_lshlrev_b32_e32 v0, 2, v0
    s_setpc_b64 s[10:11]
.Lfunc_enda:
    .size    get_3d_idx, .Lfunc_enda-get_3d_idx
                                        ; -- End function
    .section    .AMDGPU.csdata
; Device Function 
"""
    return template.splitlines(True)

def need_cal_threadidx(last_vgpr):
    vgprs_layers = [
        256,
        128,
        84,
        64,
        48,
        40,
        36,
        32,
        28,
        28,
        0
    ]
    last_vgpr += 1
    layer = 0
    for l in range(len(vgprs_layers)):
        if vgprs_layers[l] <= last_vgpr:
            layer = l-1
            break
    if vgprs_layers[layer] - last_vgpr >= 3:
        return True
    return False

def generate_caller_asm(need_cal_idx, last_vgpr, last_sgpr, lds_size, stack_size, 
    use_block_x, use_block_y, use_block_z,
    caller_name, launch_params):
# Loop States:
# [0:16]   : s[0:3] stack_base
# [16:24]  : params
# [24:32]  : loop address
# [32:96]  : s[9] stack_offset
# [96:100]  : current_task_idx
# [100:104] : step len
# [104:108]: total_tasks
# 

# sgprs:
# 0-3 stack_base
# 4-5 params
# 6-7 loop address
# 8 stack offset

# Steps:
# 1. save s[0:3], s[9] (s[7] at the beginning)
# 2. check cu_id, layer_id
# 3. init and save current_idx
# 4. init and save step=layers * (cu_upper - cu_lower)
# 5. save total_task
# 6. loop begin:
# 7.     load current_idx, step, total_task
# 8.     check current_idx and total_task
# 9.     save current_idx + step
#10.     calculate blockIdx
#11.     calculate threadIdx
#12.     prepare registers, and jump to the kernel

    need_stack = ""
    if stack_size == 0:
        need_stack = ";"

    cal_threadidx = ""
    save_threadidx = ";"
    if not need_cal_idx:
        save_threadidx = ""
        cal_threadidx = ";"        

    sgpr_idx = 6
    block_x_sgpr = 6
    block_y_sgpr = 6
    block_z_sgpr = 6
    stack_offset_sgpr = 6
    if use_block_x:
        sgpr_idx += 1
    block_y_sgpr = sgpr_idx
    if use_block_y:
        sgpr_idx += 1
    block_z_sgpr = sgpr_idx
    if use_block_z:
        sgpr_idx += 1
    stack_offset_sgpr = sgpr_idx

    template = """
;;# header start
;;# save stack_base
{need_stack}    s_mov_b32 s{stack_base0}, s0
{need_stack}    s_mov_b32 s{stack_base1}, s1
{need_stack}    s_mov_b32 s{stack_base2}, s2
{need_stack}    s_mov_b32 s{stack_base3}, s3
{need_stack}    s_mov_b32 s{stack_offset}, s7
;;# save threadIdx.x to the last vgpr
    v_mov_b32 v{threadIdx0}, v0
;;# header end
;;# calculate cu_id
    s_load_dword s2, s[4:5], 0x40
    s_mul_hi_u32 s0, s6, 0x88888889
    s_lshr_b32 s3, s0, 5
    s_mul_i32 s0, s3, 60
    s_sub_i32 s6, s6, s0
    s_waitcnt lgkmcnt(0)
    s_cmp_lt_u32 s6, s2
    s_cbranch_scc1 {caller_name}1_2
;;# load right parts
    s_load_dwordx4 s[12:15], s[4:5], 0x28
    s_load_dwordx2 s[18:19], s[4:5], 0x38
    s_mov_b32 s4, s2
    s_mov_b32 s2, 60
    s_branch {caller_name}1_3
;;# load left parts
{caller_name}1_2:
    s_load_dwordx4 s[12:15], s[4:5], 0x8
    s_load_dwordx2 s[18:19], s[4:5], 0x18
    s_mov_b32 s4, 0
{caller_name}1_3:
    v_mov_b32_e32 v1, s4
    v_cmp_lt_u32_e32 vcc, s6, v1
    v_mov_b32_e32 v1, s2
    v_cmp_ge_u32_e64 s[0:1], s6, v1
    s_waitcnt lgkmcnt(0)
    v_mov_b32_e32 v1, s12
    s_or_b64 s[0:1], vcc, s[0:1]
    v_cmp_ge_i32_e32 vcc, s3, v1
    s_or_b64 s[0:1], s[0:1], vcc
    s_and_b64 vcc, exec, s[0:1]
;;# if cu_id < cu_lower || cu_id >= cu_upper || layer_idx >= layers return;
    s_cbranch_vccnz {caller_name}1_8

;;# if threadIdx.x >= total_threads return;
;    v_mov_b32_e32 v1, {total_threads_per_block}
;    v_cmp_ge_u32_e32 vcc, v0, v1
;    s_cbranch_vccnz {caller_name}1_8

;;# init current_idx = (cu_upper - cu_lower) * layer_idx + (cu_id - cu_lower) + task_offset
;;# so far, v0=threadIdx.x, s6=cu_id, s2=cu_upper, s3=layer_idx, s4=cu_lower, s[12:14]=(layers, task_num, task_offset), s[18:19]=args
    s_sub_i32 s1, s2, s4 ;; s1 = cu_upper - cu_lower
    s_sub_i32 s5, s6, s4 ;; s5 = cu_id - cu_lower
    s_mul_i32 s0, s1, s3
    s_add_i32 s0, s0, s5
    s_add_i32 s0, s0, s14 ;; s0 = current_idx
;;# init step_len = layers * (cu_upper - cu_lower)
    s_mul_i32 s5, s1, s12 ;; s5 = step_len
;;# init total_tasks
    s_add_i32 s7, s13, s14 ;; s7 = total_task

;;# store the values
    s_mov_b32 s{current_idx}, s0
    s_mov_b32 s{step_len}, s5
    s_mov_b32 s{total_task}, s7

;;# store loop address and params
    s_getpc_b64 s[10:11]
    s_add_u32 s10, s10, {caller_name}_loop@rel32@lo+4
    s_addc_u32 s11, s11, {caller_name}_loop@rel32@hi+12
    s_mov_b32 s{loop_addr0}, s10
    s_mov_b32 s{loop_addr1}, s11
    s_mov_b32 s{param_addr0}, s18
    s_mov_b32 s{param_addr1}, s19

{save_threadidx}   s_getpc_b64 s[8:9]
{save_threadidx}   s_add_u32 s8, s8, get_3d_idx_{blk_dim_x}_{blk_dim_y}_{blk_dim_z}@rel32@lo+4
{save_threadidx}   s_addc_u32 s9, s9, get_3d_idx_{blk_dim_x}_{blk_dim_y}_{blk_dim_z}@rel32@hi+12
{save_threadidx}   s_swappc_b64 s[30:31], s[8:9]
{save_threadidx}   v_mov_b32 v{threadIdx0}, v0
{save_threadidx}   v_mov_b32 v{threadIdx1}, v1
{save_threadidx}   v_mov_b32 v{threadIdx2}, v2
;;# LOOP BEGIN:
{caller_name}_loop:
    ;; load current_idx, step, total_task
    s_mov_b32 s0, s{current_idx}
    s_mov_b32 s1, s{step_len}
    s_mov_b32 s2, s{total_task}
    s_cmp_ge_u32 s0, s2 ;; if task_idx >= total_tasks
    s_cbranch_scc1 {caller_name}1_8

    ;; update next_idx = current_idx + step
    s_add_u32 s{current_idx}, s{current_idx}, s{step_len}

;;  call get_3d_idx for blockIdx
;;  Input: v0
;;  Output: v0, v1, v2
    v_mov_b32 v0, s0
    s_getpc_b64 s[8:9]
    s_add_u32 s8, s8, get_3d_idx_{grid_dim_x}_{grid_dim_y}_{grid_dim_z}@rel32@lo+4
    s_addc_u32 s9, s9, get_3d_idx_{grid_dim_x}_{grid_dim_y}_{grid_dim_z}@rel32@hi+12
    s_swappc_b64 s[30:31], s[8:9]

;;  Save blockIdx in s12, s13, s14

    v_readfirstlane_b32 s12, v0
    v_readfirstlane_b32 s13, v1
    v_readfirstlane_b32 s14, v2

;;  call get_3d_idx for threadIdx
;;  Input: v0
;;  Output: v0, v1, v2
{cal_threadidx}    v_mov_b32 v0, v{last_vgpr}
{cal_threadidx}    s_getpc_b64 s[8:9]
{cal_threadidx}    s_add_u32 s8, s8, get_3d_idx_{blk_dim_x}_{blk_dim_y}_{blk_dim_z}@rel32@lo+4
{cal_threadidx}    s_addc_u32 s9, s9, get_3d_idx_{blk_dim_x}_{blk_dim_y}_{blk_dim_z}@rel32@hi+12
{cal_threadidx}    s_swappc_b64 s[30:31], s[8:9]

;;# footer begin
;;# set s[4:5] to params
    s_mov_b32 s4, s{param_addr0}
    s_mov_b32 s5, s{param_addr1}
;;# set s[6:8] to task_idx
    s_mov_b32 s{block_x_sgpr}, s12
    s_mov_b32 s{block_y_sgpr}, s13
    s_mov_b32 s{block_z_sgpr}, s14
;;# set v[0:2] to threadIdx
{save_threadidx}    v_mov_b32 v0, v{threadIdx0}
{save_threadidx}    v_mov_b32 v1, v{threadIdx1}
{save_threadidx}    v_mov_b32 v2, v{threadIdx2}
;;# load init states
{need_stack}    s_mov_b32 s0, s{stack_base0}
{need_stack}    s_mov_b32 s1, s{stack_base1}
{need_stack}    s_mov_b32 s2, s{stack_base2}
{need_stack}    s_mov_b32 s3, s{stack_base3}
{need_stack}    s_mov_b32 s{stack_offset_sgpr}, s{stack_offset}
;;# jump to func
    s_getpc_b64 s[10:11]
    s_add_u32 s10, s10, {caller_name}_device_wrapper@rel32@lo+4
    s_addc_u32 s11, s11, {caller_name}_device_wrapper@rel32@hi+12
    s_setpc_b64 s[10:11]
;;# footer end
{caller_name}1_8:
    s_endpgm
""".format(last_vgpr=last_vgpr, lds_size=lds_size, caller_name=caller_name,
    grid_dim_x=launch_params[0], grid_dim_y=launch_params[1], grid_dim_z=launch_params[2],
    blk_dim_x=launch_params[3], blk_dim_y=launch_params[4], blk_dim_z=launch_params[5],
    total_threads_per_block=launch_params[3]*launch_params[4]*launch_params[5],
    param_addr0=last_sgpr+0, param_addr1=last_sgpr+1, loop_addr0=last_sgpr+2, loop_addr1=last_sgpr+3,
    current_idx=last_sgpr+4, step_len=last_sgpr+5, total_task=last_sgpr+6,
    stack_base0 = last_sgpr+7, stack_base1 = last_sgpr+8, stack_base2 = last_sgpr+9, stack_base3 = last_sgpr+10,
    stack_offset = last_sgpr+11,
    threadIdx0=last_vgpr+0,threadIdx1=last_vgpr+1,threadIdx2=last_vgpr+2,
    need_stack=need_stack,save_threadidx=save_threadidx,cal_threadidx=cal_threadidx,
    block_x_sgpr=block_x_sgpr, block_y_sgpr=block_y_sgpr,block_z_sgpr=block_z_sgpr,
    stack_offset_sgpr=stack_offset_sgpr
)
    return template

def generate_header_asm(last_vgpr, lds_size):
    template = """
;;# header start
    v_mov_b32_e32 v1, %d
    v_mov_b32 v2, s0
    ds_write_b32 v1, v2 offset:0
    v_mov_b32 v3, s1
    ds_write_b32 v1, v3 offset:4
    v_mov_b32 v4, s2
    ds_write_b32 v1, v4 offset:8
    v_mov_b32 v5, s3
    ds_write_b32 v1, v5 offset:12
    v_mov_b32 v6, s4 
    ds_write_b32 v1, v6 offset:16
    v_mov_b32 v7, s5
    ds_write_b32 v1, v7 offset:20
    v_mov_b32 v8, s6
    ds_write_b32 v1, v8 offset:24
;;# s7 is related to the wavefront
;;# v10 = (threadIdx.x / 64) * 4 = threadIdx.x >> 4
;;# save s7 to sm[28 + v1]
;;# from 28 -> 92
    v_mov_b32_e32 v10, v0
    v_lshrrev_b32 v10, 6, v10
    v_lshlrev_b32 v10, 2, v10
    v_add_u32 v10, v10, v1
    v_mov_b32 v9, s7
    ds_write_b32 v10, v9 offset:28
    s_waitcnt lgkmcnt(0)
;;# save threadIdx.x to the last vgpr
    v_mov_b32 v%d, v0
;;# save num_iterations
    v_mov_b32 v2, 0
    ds_write_b32  v1, v2 offset:92
    s_waitcnt lgkmcnt(0)
;;# save pc
    s_getpc_b64 s[8:9]
    v_mov_b32_e32 v1, %d
    v_mov_b32 v2, s8
    v_mov_b32 v3, s9
    ds_write_b32 v1, v2 offset:96
    ds_write_b32 v1, v3 offset:100
    s_waitcnt lgkmcnt(0)
    s_barrier
;;# header end\n
""" % (lds_size, last_vgpr, lds_size)
    return template

def generate_footer_asm(last_vgpr, last_sgpr, lds_size):
    template = """
;;# footer begin
    s_barrier
;;# load return address
    s_mov_b32 s8, s{loop_addr0}
    s_mov_b32 s9, s{loop_addr1}
    s_setpc_b64 s[8:9]
;;# footer end\n
""".format(
    lds_size = lds_size, last_vgpr = last_vgpr,
    loop_addr0=last_sgpr+2, loop_addr1=last_sgpr+3
)
    return template


if len(sys.argv) != 3:
    print("Usage: python3 generate_asm_loop.py kernel.s schedule.json")
    exit(1)

asm_file = open(sys.argv[1], "r")
lines = asm_file.readlines()
asm_file.close()

schedule = json.loads(open(sys.argv[2], "r").read())


funcs = {}

for func in schedule["kernels"]:
    funcs[func["name"]] = func

# Steps:
# Round #1: count vgpr/sgpr and lds usage
# Round #2: add header and footer assembly, vgpr ++

func_info = {}

current_func = ""

def collect_info(line, type, key, current_func, func_info):
    if line.find(type) != -1:
        value = int(line.strip().split(" ")[-1])
        if not current_func in func_info:
            func_info[current_func] = {}
        func_info[current_func][key] = value
        # print("func: %s, key: %s, value: %d" % (current_func, key, value))

## Round #1 
info_list = [
    [".amdhsa_group_segment_fixed_size", "lds"],
    [".amdhsa_next_free_vgpr", "next_free_vgpr"],
    [".amdhsa_next_free_sgpr", "next_free_sgpr"],
    [".sgpr_count:", "sgpr"],
    [".vgpr_count:", "vgpr"],
    [".amdhsa_private_segment_fixed_size", "stack"],
    [".vgpr_spill_count", "vgpr_spill"],
    [".sgpr_spill_count", "sgpr_spill"],
    [".amdhsa_system_sgpr_workgroup_id_x", "block_x"],
    [".amdhsa_system_sgpr_workgroup_id_y", "block_y"],
    [".amdhsa_system_sgpr_workgroup_id_z", "block_z"],
]

for line in lines:
    if line.find(".amdhsa_kernel") != -1:
        func_name = line.strip().split(" ")[1]
        if func_name.find("framework") == -1:
            if func_name.find("_device_wrapper") == -1:
                current_func = ""
            else:
                current_func = func_name.replace("_device_wrapper", "")
        else:
            current_func = ""
    if line.find(".name") != -1:
        func_name = line.strip().split(" ")[-1]
        if func_name.find("framework") == -1:
            if func_name.find("_device_wrapper") == -1:
                current_func = ""
            else:
                current_func = func_name.replace("_device_wrapper", "")
        else:
            current_func = ""
    if current_func != "":
        for info in info_list:
            collect_info(line, info[0], info[1], current_func, func_info)
    

for key, value in func_info.items():
    value["vgpr"] = max(value["next_free_vgpr"], value["vgpr"], 20)
    value["sgpr"] = max(value["next_free_sgpr"], value["sgpr"], 36)
    new_sgpr_count = value["sgpr"]
    if value["stack"] == 0:
        new_sgpr_count += 7
    else:
        new_sgpr_count += 12
    value["new_sgpr"] = new_sgpr_count
    if need_cal_threadidx(value["vgpr"]):
        value["new_vgpr"] = value["vgpr"] + 1
        value["need_cal_idx"] = True
    else:
        value["new_vgpr"] = value["vgpr"] + 3
        value["need_cal_idx"] = False
# print(func_info)

## Round #2
new_lines = []
current_func = ""

for line in lines:
    if line.find(".globl") != -1:
        func_name = line.strip().split("	")[1]
        if func_name.find("framework") == -1:
            if func_name.find("_device_wrapper") == -1:
                current_func = func_name
            else:
                current_func = func_name.replace("_device_wrapper", "")
            if current_func not in funcs:
                current_func = ""
        else:
            current_func = ""
    # if line.find("; %bb.0:") != -1 and current_func != "":
    #     new_lines.extend(generate_header_asm(func_info[current_func]["vgpr"], func_info[current_func]["lds"]).splitlines(True))
    if line.find(";; end_flag") != -1 and current_func != "":
        new_lines.extend(generate_footer_asm(
            func_info[current_func]["vgpr"], func_info[current_func]["sgpr"], func_info[current_func]["lds"]).splitlines(True))
    if line.find(";; caller_flag") != -1 and current_func != "":
        new_lines.extend(generate_caller_asm(
            func_info[current_func]["need_cal_idx"],
            func_info[current_func]["vgpr"],
            func_info[current_func]["sgpr"],
            func_info[current_func]["lds"], 
            func_info[current_func]["stack"],
            func_info[current_func]["block_x"],
            func_info[current_func]["block_y"],
            func_info[current_func]["block_z"],
            current_func, funcs[current_func]["launch_params"]).splitlines(True))
    if line.find(".amdhsa_next_free_vgpr") != -1 and current_func != "":
        new_lines.append("        .amdhsa_next_free_vgpr %d\n" % (func_info[current_func]["new_vgpr"]))
        continue
    if line.find(".amdhsa_next_free_sgpr") != -1 and current_func != "":
        new_lines.append("        .amdhsa_next_free_sgpr %d\n" % (func_info[current_func]["new_sgpr"]))
        continue
    if line.find(".amdhsa_group_segment_fixed_size") != -1 and current_func != "":
        new_lines.append("        .amdhsa_group_segment_fixed_size %d\n" % (func_info[current_func]["lds"]))
        continue
    if line.find(".amdhsa_private_segment_fixed_size") != -1 and current_func != "":
        new_lines.append("        .amdhsa_private_segment_fixed_size %d\n" % (func_info[current_func]["stack"]))
        # print(current_func)
        # print(func_info[current_func]["stack"])
        continue
    if line.find(".amdhsa_system_sgpr_private_segment_wavefront_offset") != -1 and current_func != "":
        value = 0
        if func_info[current_func]["stack"] > 0:
            value = 1
        new_lines.append("      .amdhsa_system_sgpr_private_segment_wavefront_offset %d\n" % (value))
        continue

    new_lines.append(line)


# sort the kernels
kernels = []
for line in lines:
    if line.find(".name") != -1:
        func_name = line.strip().split(" ")[-1]
        kernels.append(func_name)

# print(kernels)
# print(func_info)

lines = new_lines

new_lines = []

current_func_idx = -1
for line in lines:
    if line.find("- .args:") != -1:
        current_func_idx += 1
        current_func = kernels[current_func_idx]
        if current_func.find("_device_wrapper") != -1:
            current_func = current_func.replace("_device_wrapper", "")
        if current_func not in func_info:
            current_func = ""
    if line.find(".vgpr_count:") != -1 and current_func != "":
        new_lines.append("    .vgpr_count:     %d\n" % (func_info[current_func]["new_vgpr"]))  
        continue
    if line.find(".sgpr_count:") != -1 and current_func != "":
        new_lines.append("    .sgpr_count:     %d\n" % (func_info[current_func]["new_sgpr"]))  
        continue   
    if line.find(".group_segment_fixed_size:") != -1 and current_func != "":
        new_lines.append("    .group_segment_fixed_size: %d\n" % (func_info[current_func]["lds"]))
        continue
    if line.find(".private_segment_fixed_size:") != -1 and current_func != "":
        new_lines.append("    .private_segment_fixed_size:     %d\n" % (func_info[current_func]["stack"]))
        continue
    if line.find(".vgpr_spill_count:") != -1 and current_func != "":
        new_lines.append("    .vgpr_spill_count: %d\n" % (func_info[current_func]["vgpr_spill"]))
        continue
    if line.find(".sgpr_spill_count:") != -1 and current_func != "":
        new_lines.append("    .sgpr_spill_count: %d\n" % (func_info[current_func]["sgpr_spill"]))
        continue
    new_lines.append(line)

# final = generate_device_func_header()
final = []
final.extend(new_lines)
# print(new_lines)

asm_file = open(sys.argv[1], "w")
asm_file.writelines(final)
asm_file.close()
