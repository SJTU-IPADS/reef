import os
import sys
import json

#############################################################################################################################
# 
# This script is used to transform TVM generated GPU code.                                                                  
#                                                                                                                           
# For each kernel K, the transformation consists of X steps:                                                                
#   1. K will be replaced by a __device__ function with name K_device                                                       
#   2. blockIdx will be replaced by task_idx                                                                                
#   3. threadIdx will be replaced by thread_idx                                                                             
#   4. __shared__ variable will be replaced by a pointer to (the base address shared memory + offset)                       
#   5. TODO: count __syncthreads() and deal with early-exit threads                                                         
#   6. K_device will append new parameters, such as "shared_memory", "task_idx", "thread_idx", "task_dim", "thread_dim"     
#   7. A new __device__ __noinline__ function K will be generated using persistant threads scheme                           
#   8. A __global__ function K_wrapper will be generated for calculating the register usage                                 
#   9. An example kernel merge will be generated                                                                            
#                                                                                                                           
#############################################################################################################################

def split_function_declaration(line):
    parts = line.split("void")
    right_parts = parts[1].split("(")
    name = right_parts[0].strip()
    parameters_str = right_parts[1].split(")")[0]
    return_type = "void"
    header = parts[0]
    parameter_str_list = parameters_str.split(", ")
    parameters = []
    for param_str in parameter_str_list:
        parts = param_str.split(" ")
        param_name = parts[-1]
        param_type = " ".join(parts[:-1])
        parameters.append({"name": param_name, "type": param_type})
    return header, return_type, name, parameters

def find_all_func_params(lines):
    funcs = {}
    for line in lines:
        if line.find("__global__") != -1:
            header, return_type, func_name, func_params = split_function_declaration(line)
            funcs[func_name] = func_params
    return funcs

def find_all_func_attr(lines):
    funcs = {}
    for line in lines:
        if line.find("__global__") != -1:
            header, _, func_name, _ = split_function_declaration(line)
            parts = header.split("__global__")
            right_part = parts[-1].strip()
            funcs[func_name] = right_part
    return funcs

def find_all_sgpr_usage(lines):
    funcs = {}
    current_func = ""
    for line in lines:
        # replace amdhas_next_free_vgpr
        if line.find(".name:") != -1:
            kernel_name = line.strip().split(" ")[-1]
            current_func = kernel_name
        elif line.find(".sgpr_count:") != -1:
            value = int(line.strip().split(" ")[-1])
            if value >= 90:
                value = 89
            if value < 30:
                value = 30
            funcs[current_func] = value
    return funcs
    
def find_all_vgpr_usage(lines):
    funcs = {}
    current_func = ""
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
    for line in lines:
            # replace amdhas_next_free_vgpr
        if line.find(".name:") != -1:
            kernel_name = line.strip().split(" ")[-1]
            current_func = kernel_name
        elif line.find(".vgpr_count:") != -1:
            value = int(line.strip().split(" ")[-1])
            layer = 0
            for i in range(len(vgprs_layers)):
                if value > vgprs_layers[i]:
                    layer = i-1
                    break
            if vgprs_layers[layer] - value >= 3:
                value = vgprs_layers[layer] - 3
            else:
                value = vgprs_layers[layer] - 1
            funcs[current_func] = value

    return funcs   


def generate_function_declaration(return_type, name, params):
    params_str_list = []
    for param in params:
        param_str = param["type"] + " " + param["name"]
        params_str_list.append(param_str)
    return return_type + " " + name + "(" + ", ".join(params_str_list) + ")"


def replace_global_with_device(lines):
    new_lines = []
    for line in lines:
        if line.find("void") != -1:
            header, return_type, func_name, func_params = split_function_declaration(line)
            new_line = "__device__ " + generate_function_declaration(return_type, func_name + "_device", func_params)
            if line.find("{") != -1:
                new_line = new_line + "{"
            new_line = new_line + "\n"
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    return new_lines

def replace_blockIdx_with_task_idx(lines):
    new_lines = []
    for line in lines:
        # if line.find("void") != -1:
        #     # insert new parameter to the function
        #     header, return_type, func_name, func_params = split_function_declaration(line)
        #     func_params.insert(0, {"name": "task_idx", "type": "dim3"})
        #     new_line = header + " " + generate_function_declaration(return_type, func_name, func_params)
        #     if line.find("{") != -1:
        #         new_line = new_line + "{"
        #     new_line = new_line + "\n"
        #     new_lines.append(new_line)
        # else:
            # new_lines.append(line.replace("blockIdx", "task_idx"))
        new_lines.append(line.replace("blockIdx", "task_idx"))
    return new_lines

def replace_threadIdx_with_thread_idx(lines):
    new_lines = []
    for line in lines:
        # if line.find("void") != -1:
        #     # insert new parameter to the function
        #     header, return_type, func_name, func_params = split_function_declaration(line)
        #     func_params.insert(0, {"name": "thread_idx", "type": "dim3"})
        #     new_line = header + " " + generate_function_declaration(return_type, func_name, func_params)
        #     if line.find("{") != -1:
        #         new_line = new_line + "{"
        #     new_line = new_line + "\n"
        #     new_lines.append(new_line)
        # else:
        #     new_lines.append(line.replace("threadIdx", "thread_idx"))
        new_lines.append(line.replace("threadIdx", "thread_idx"))
    return new_lines

def replace_shared_memory(lines):
    new_lines = []
    current_function = ""
    current_offset = 0
    for line in lines:
        if line.find("void") != -1:
            # header, return_type, func_name, func_params = split_function_declaration(line)
            # current_function = func_name
            # current_offset = 0
            # func_params.insert(0, {"name": "shared_memory", "type": "char*"})
            # new_line = header + " " + generate_function_declaration(return_type, func_name, func_params)
            # if line.find("{") != -1:
            #     new_line = new_line + "{"
            # new_line = new_line + "\n"
            current_offset = 0
            new_lines.append(line)
        elif line.find("__shared__") != -1:
            parts = line.strip().split(" ")
            var_type = parts[1]
            var = parts[2]
            print(parts)
            var_name = var.split("[")[0]
            var_size = var.split("[")[1].split("]")[0]
            if var_type != "float":
                print("Unsupported shared memory type: " + var_type)
                exit(0)
            new_line = "  float* " + var_name + " = " + "(float*)(shared_memory+" + str(current_offset) + ");\n"
            current_offset = current_offset + int(var_size) * 4
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    return new_lines

def add_device_function_param(lines):
    new_lines = []
    for line in lines:
        if line.find("void") != -1:
            # insert new parameter to the function
            header, return_type, func_name, func_params = split_function_declaration(line)
            
            func_params.insert(0, {"name": "thread_idx", "type": "dim3"})
            func_params.insert(0, {"name": "task_idx", "type": "dim3"})
            # func_params.insert(0, {"name": "shared_memory", "type": "char*"})

            new_line = header + " " + generate_function_declaration(return_type, func_name, func_params)
            if line.find("{") != -1:
                new_line = new_line + "{"
            new_line = new_line + "\n"
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    return new_lines

def add_global_definition(lines):
    new_lines = []
    insert_line = 0
    for i in range(len(lines)):
        if lines[i].find("void") != -1:
            insert_line = i
            break
    new_lines.extend(lines[:insert_line])
    new_lines.extend("""#define CU_NUM 60

__device__ __forceinline__ bool is_first_thread() {
  return threadIdx.x == 0;
}

__device__ __forceinline__ unsigned int get_cu_id() {
  return blockIdx.x % CU_NUM;
}

__device__ __forceinline__ dim3 get_3d_idx(int idx, dim3 dim) {
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
}

""".splitlines(True))
    new_lines.extend(lines[insert_line:])
    return new_lines

def generate_vgpr_num(occupancy_hint):
    # __attribute__((amdgpu_waves_per_eu(5,5)))
    if (occupancy_hint.find("__attribute__") == -1):
        return 0
    occupancy = int(occupancy_hint.split(",")[-1].split(")")[0])
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
    return vgprs_layers[occupancy] - 1

def generate_sgpr_num(occupancy_hint):
    if (occupancy_hint.find("__attribute__") == -1):
        return 0
    occupancy = int(occupancy_hint.split(",")[-1].split(")")[0])
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
    return sgprs_layers[occupancy]



def generate_func_attribute(func_attr, default_vgpr, default_sgpr):
    attrs = func_attr.strip().split(" ")
    new_attrs = []
    vgpr_num = -1
    sgpr_num = -1
    for attr in attrs:
        if attr.find("amdgpu_waves_per_eu") != -1:
            if vgpr_num == -1:
                vgpr_num = generate_vgpr_num(attr)
            if sgpr_num == -1:
                sgpr_num = generate_sgpr_num(attr)
            continue
        if attr.find("amdgpu_num_vgpr") != -1:
            vgpr_num = 1
        if attr.find("amdgpu_num_sgpr") != -1:
            sgpr_num = 1
        new_attrs.append(attr)
    if vgpr_num < 0:
        new_attrs.append("__attribute__((amdgpu_num_vgpr(%d)))" % (default_vgpr))
    if sgpr_num < 0:
        new_attrs.append("__attribute__((amdgpu_num_sgpr(%d)))" % (default_sgpr))
    
    return " ".join(new_attrs)


def generate_device_wrapper(func_params, func_attrs, launch_params, func_vgprs, func_sgprs):
    result = []
    for func_name, func_param in func_params.items():
        launch_param = [1, 1, 1, 1, 1, 1]
        if func_name.strip() in launch_params:
            launch_param = launch_params[func_name.strip()]
        param_names = []
        for param in func_param:
            param_names.append("(%s)%s"%(param["type"], param["name"]))
        func_def = generate_function_declaration("void", func_name + "_device_wrapper", func_param)
        func_call = func_name + "_device(" + ",".join(param_names) +")"
        func_template = """
extern "C" __global__ {attr} {func_def} {left}
    // Force the compiler to use all the index
    if (threadIdx.x + threadIdx.y * {block_dim_x} + threadIdx.z * {block_dim_y} * {block_dim_x} >= {block_dim_x} * {block_dim_y} * {block_dim_z}) return;
    // if (blockIdx.x + blockIdx.y * {grid_dim_x} + blockIdx.z * {grid_dim_y} * {grid_dim_x} >= {grid_dim_x} * {grid_dim_y} * {grid_dim_z}) return;
    {func_call};
    asm volatile(";; end_flag"); // jump back to the caller
{right}
""".format(attr = generate_func_attribute(func_attrs[func_name], func_vgprs[func_name], func_sgprs[func_name]), func_def =func_def, 
block_dim_x = launch_param[3], block_dim_y = launch_param[4], block_dim_z = launch_param[5],
grid_dim_x = launch_param[0], grid_dim_y = launch_param[1], grid_dim_z = launch_param[2],
func_call = func_call, left="{", right="}"
)
        result.extend(func_template.splitlines(True))
    return result

def generate_global_caller(func_params):
    result = []
    for func_name, func_param in func_params.items():
        func_template = """
extern "C" __global__  void %s(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {
    asm volatile(";; caller_flag");
    return;
}
""" % (
    func_name
)
        result.extend(func_template.splitlines(True))
    return result

def generate_dim_transformations(launch_params):
    # 1. dedup
    dims = set()
    for k, params in launch_params.items():
        dims.add((params[0], params[1], params[2]))
        dims.add((params[3], params[4], params[5]))
    result = []
    for dim in dims:
        template = """
extern "C" __device__ __noinline__ dim3 get_3d_idx_{x}_{y}_{z}(int idx) {l}
  dim3 dim({x}, {y}, {z});
  dim3 result;
  result.x = idx % dim.x;
  result.y = idx / dim.x % dim.y;
  result.z = idx / (dim.x * dim.y);
  return result;
{r}
""".format(x = dim[0], y = dim[1], z = dim[2], l = "{", r = "}")
        result.extend(template.splitlines(True))
    call_kernel = """
__global__ void get_3d_idx_caller(int* buf) {
    dim3 task_idx;
"""
    for dim in dims:
        call_template = """
    task_idx = get_3d_idx_{x}_{y}_{z}(threadIdx.x);
    buf[task_idx.x] = task_idx.x;
    buf[task_idx.y] = task_idx.y;
    buf[task_idx.z] = task_idx.z;
""".format(x = dim[0], y = dim[1], z = dim[2])
        call_kernel += call_template
    call_kernel += "\n}\n"
    result.extend(call_kernel.splitlines(True))
    return result

def generate_persistent_thread(func_params, func_attrs, launch_params):
    result = []
    for func_name, func_param in func_params.items():
        launch_param = [1, 1, 1, 1, 1, 1]
        if func_name.strip() in launch_params:
            launch_param = launch_params[func_name.strip()]
        outter_func_param = [
            {"name": "task_slots", "type": "int*"},
            {"name": "task_num", "type": "int"},
            {"name": "task_offset", "type": "int"},
            {"name": "cu_lower", "type": "int"},
            {"name": "cu_upper", "type": "int"},
            # {"name": "shared_memory", "type": "char*"},
            {"name": "args", "type": "float**"}
        ]
        # outter_func_param.extend(func_param)

        func_def = generate_function_declaration("void", func_name + "_device_wrapper", outter_func_param)
        params_name = ["task_idx", "thread_idx"]
        for i in range(len(func_param)):
            params_name.append("args[%d]" %(i))
        func_call = func_name+ "_device(" + ", ".join(params_name) + ")"

        func_template = """
extern "C" __global__ __attribute__((amdgpu_num_vgpr(%d))) void %s(
    void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition) {

    int layers, task_num, task_offset, cu_upper, cu_lower;
    float** args;
    unsigned int cu_id = get_cu_id();
    if (cu_id < cu_partition) {
      layers = layers_l;
      task_num = task_num_l;
      task_offset = task_offset_l;
      args = param_l;
      cu_upper = cu_partition;
      cu_lower = 0;
    } else {
      layers = layers_r;
      task_num = task_num_r;
      task_offset = task_offset_r;
      args = param_r;
      cu_upper = 60;
      cu_lower = cu_partition;
    }
    dim3 task_dim;
    dim3 thread_dim;
    if (task_offset < 0) {
      // impossible
      dim3 *temp = (dim3*)func_l;
      task_dim = dim3(temp->x, temp->y, temp->z);
      thread_dim = dim3(temp->x, temp->y, temp->z);
    } else {
      task_dim = dim3(%d,%d,%d);
      thread_dim = dim3(%d,%d,%d);
    }

    int layer_idx = blockIdx.x / CU_NUM; 
    if (threadIdx.x >= thread_dim.x * thread_dim.y * thread_dim.z) return;
    if (cu_id < cu_lower || cu_id >= cu_upper) return;
    if (layer_idx >= layers) return;

    dim3 thread_idx = get_3d_idx(threadIdx.x, thread_dim);
    
    HIP_DYNAMIC_SHARED(char, shared_memory);
    int num_iteration;
    num_iteration = *(int *)(shared_memory + 92);
    __syncthreads();
    if (is_first_thread()) 
      *(int *)(shared_memory  + 92) = num_iteration + 1;
    __syncthreads();


    unsigned int idx = (cu_upper - cu_lower) * layer_idx + (cu_id - cu_lower) + task_offset;
    unsigned int total_task = task_offset + task_num;
    idx += num_iteration * layers * (cu_upper - cu_lower);
    if (idx >= total_task) return;
    dim3 task_idx = get_3d_idx(idx, task_dim);
    __syncthreads();
    %s;
    
    asm volatile(";; end_flag");
}
""" % (generate_vgpr_hint(func_attrs[func_name]), func_name, 
launch_param[0], launch_param[1], launch_param[2],
launch_param[3], launch_param[4], launch_param[5],
func_call
)
        result.extend(func_template.splitlines(True))
    return result


def generate_global_wrappers(func_params):
    func_calls = []
    for func_name, param in func_params.items():
        param_list = [
            "task_dim",
            "thread_dim",
            "task_slots",
            "task_num",
            "task_offset",
            "cu_lower",
            "cu_upper",
            # "shared_memory",
            "args"
        ]
        # for i in range(len(param)):
        #     param_list.append("p" + str(i))

        func_call = "%s(%s);" % (func_name + "_device_wrapper", ", ".join(param_list))
        func_calls.append(func_call)

    outter_func_param = [
        {"name": "task_dim", "type": "dim3"},
        {"name": "thread_dim", "type": "dim3"},
        {"name": "task_slots", "type": "int*"},
        {"name": "task_num", "type": "int"},
        {"name": "task_offset", "type": "int"},
        {"name": "cu_lower", "type": "int"},
        {"name": "cu_upper", "type": "int"},
        # {"name": "shared_memory", "type": "char*"},
        {"name": "args", "type": "float**"},
    ]
    func_decl = "\n__global__ " + generate_function_declaration("void", "global_kernel", outter_func_param) + "{\n"
    result = [func_decl]
    for func_call in func_calls:
        result.append("  " + func_call + "\n")
    result.append("}\n")
    return result

def generate_merged_kernel(func_params):
# Example:
# __global__ void merged_kernel(
#     dim3 task_dim_l, dim3 thread_dim_l, int task_num_l, int task_offset_l, float** param_l,
#     dim3 task_dim_r, dim3 thread_dim_r, int task_num_r, int task_offset_r, float** param_r,
#     int cu_partition, int* task_slots) 
# {
#   int cu_id = get_cu_id();
#   extern __shared__ char shared_memory[];
#   if (cu_id < cu_partition) {
#     fused_nn_contrib_conv2d_winograd_without_weight_transform_add_nn_relu_kernel2_device_wrapper(task_dim_l, thread_dim_l, task_slots, task_num_l, task_offset_l, 0, cu_partition, shared_memory, param_l);
#   } else {
#     fused_nn_conv2d_3_kernel0_device_wrapper(task_dim_r, thread_dim_r, task_slots, task_num_r, task_offset_r, cu_partition, CU_NUM, shared_memory, param_r);
#   }
# }
    func_l = ""
    func_r = ""
    for func_name, func_param in func_params.items():
        if func_l == "":
            func_l = func_name
            continue
        if func_r == "":
            func_r = func_name
            break

    return ("""
extern "C" __global__ void merged_kernel(
    void* func_l, dim3 task_dim_l, dim3 thread_dim_l, int task_num_l, int task_offset_l, float** param_l,
    void* func_r, dim3 task_dim_r, dim3 thread_dim_r, int task_num_r, int task_offset_r, float** param_r,
    int cu_partition, int* task_slots) 
{
  int cu_id = get_cu_id();
  HIP_DYNAMIC_SHARED(char, shared_memory);
  if (cu_id < cu_partition) {
    %s_device_wrapper(task_dim_l, thread_dim_l, task_slots, task_num_l, task_offset_l, 0, cu_partition, shared_memory, param_l);
  } else {
    %s_device_wrapper(task_dim_r, thread_dim_r, task_slots, task_num_r, task_offset_r, cu_partition, CU_NUM, shared_memory, param_r);
  }
}

""" % (func_l, func_r)).splitlines(True)

def generate_extern_kernel(func_params):
#     result = []
   
#     outter_func_param = [
#         {"name": "func_l", "type": "void*"},
#         {"name": "task_dim_l", "type": "dim3"},
#         {"name": "thread_dim_l", "type": "dim3"},
#         {"name": "task_num_l", "type": "int"},
#         {"name": "task_offset_l", "type": "int"},
#         {"name": "param_l", "type": "float**"},
#         {"name": "func_r", "type": "void*"},
#         {"name": "task_dim_r", "type": "dim3"},
#         {"name": "thread_dim_r", "type": "dim3"},
#         {"name": "task_num_r", "type": "int"},
#         {"name": "task_offset_r", "type": "int"},
#         {"name": "param_r", "type": "float**"},
#         {"name": "cu_partition", "type": "int"},
#         {"name": "task_slots", "type": "int*"},
#     ]

#     func_call_param =  [
#         "task_dim",
#         "thread_dim",
#         "task_slots",
#         "task_num",
#         "task_offset",
#         "cu_lower",
#         "cu_upper",
#         "shared_memory",
#         "args",
#     ]
#     func_call_param_str = ", ".join(func_call_param)
    
#     for func_name, func_param in func_params.items():
#         func_decl = generate_function_declaration("void", func_name, outter_func_param)
    
#         template = """
# extern "C" __global__ %s {
#     HIP_DYNAMIC_SHARED(char, shared_memory);
#     %s_device_wrapper(%s);
# }

# """ % (func_decl, func_name, func_call_param_str)
#         result.extend(template.splitlines(True))

    result = ""

    for func_name, func_param in func_params.items():
        result += """
extern "C" __global__ void %s(  
  void* func_l, dim3 task_dim_l, dim3 thread_dim_l, int task_num_l, int task_offset_l, float** param_l,
  void* func_r, dim3 task_dim_r, dim3 thread_dim_r, int task_num_r, int task_offset_r, float** param_r,
  int cu_partition, int* task_slots) {
    dim3 task_dim, thread_dim;
    int task_num, task_offset, cu_upper, cu_lower;
    float** args;
    unsigned int cu_id = get_cu_id();
    if (cu_id < cu_partition) {
      task_dim = task_dim_l;
      thread_dim = thread_dim_l;
      task_num = task_num_l;
      task_offset = task_offset_l;
      args = param_l;
      cu_upper = cu_partition;
      cu_lower = 0;
    } else {
      task_dim = task_dim_r;
      thread_dim = thread_dim_r;
      task_num = task_num_r;
      task_offset = task_offset_r;
      args = param_r;
      cu_upper = 60;
      cu_lower = cu_partition;
    }
    %s_device_wrapper(task_dim, thread_dim, task_slots, task_num, task_offset, cu_lower, cu_upper, args);
}

""" %(func_name, func_name)
    return result.splitlines(True)


def generate_get_address_kernel(func_params):
    
    result = """
extern "C" __global__ void get_function_address(long long int* address) {
 
"""
    func_names = list(func_params.keys())

    func_names = sorted(func_names)
    i = 0
    for func_name in func_names:
        result += "  address[%d] = (long long int)(%s_device_wrapper);\n" % (i, func_name)
        i += 1
    result += "}\n"
    return result.splitlines(True)

def generate_call_framework():
    result = ""
    result += """
#define CALL_FRAMEWORK(idx) \\
extern "C" __global__ void call_framework_##idx(\\
  void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,\\
  void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,\\
  int cu_partition) \\
{\\
  asm volatile(\\
    "  s_load_dwordx2 s[14:15], s[4:5], 0x0\\n"\\
    "  s_waitcnt lgkmcnt(0)\\n"\\
    "  s_setpc_b64 s[14:15]\\n"\\
    "  s_endpgm\\n"\\
  );\\
}

"""
    for i in range(1, 11):
        result += "CALL_FRAMEWORK(%d)\n" % (i)
    return result.splitlines(True)

def generate_merge_framework():
    result = """
#define MERGE_FRAMEWORK(idx) \\
extern "C" __global__ void merge_framework_##idx(\\
  void* func_l, int layers_l, int task_num_l, int task_offset_l, float** param_l,\\
  void* func_r, int layers_r, int task_num_r, int task_offset_r, float** param_r,\\
  int cu_partition) \\
{\\
  asm volatile(\\
    "  s_load_dword s10, s[4:5], 0x40\\n"\\
    "  s_load_dwordx2 s[12:13], s[4:5], 0x0\\n"\\
    "  s_load_dwordx2 s[14:15], s[4:5], 0x20\\n"\\
    "  s_mul_hi_u32 s11, s6, 0x88888889\\n"\\
    "  s_lshr_b32 s11, s11, 5\\n"\\
    "  s_mul_i32 s11, s11, 60\\n"\\
    "  s_sub_i32 s11, s6, s11\\n"\\
    "  s_waitcnt lgkmcnt(0)\\n"\\
    "  s_cmp_ge_u32 s11, s10\\n"\\
    "  s_mov_b64 s[10:11], -1\\n"\\
    "  s_cbranch_scc1 MyBB"#idx"_3\\n"\\
    "; %bb.1:                                ; %Flow\\n"\\
    "  s_andn2_b64 vcc, exec, s[10:11]\\n"\\
    "  s_cbranch_vccz MyBB"#idx"_4\\n"\\
    "  s_endpgm\\n"\\
    "MyBB"#idx"_3:\\n"\\
    "  s_setpc_b64 s[14:15]\\n"\\
    "  s_endpgm\\n"\\
    "MyBB"#idx"_4:\\n"\\
    "  s_setpc_b64 s[12:13]\\n"\\
    "  s_endpgm\\n"\\
  );\\
}
"""
    for i in range(1, 11):
        result += "MERGE_FRAMEWORK(%d)\n" % (i)
    for i in range(1, 11):
        result += "MERGE_FRAMEWORK(nostack_%d)\n" % (i)
    
    return result.splitlines(True)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python transform_kernel.py source_code.cu schedule.json source_code_asm.s")
        exit(0)

    f = open(sys.argv[1], "r")
    lines = f.readlines()
    f.close()

    schedule = json.loads(open(sys.argv[2], "r").read())
    asm = open(sys.argv[3], "r").readlines()

    func_params = find_all_func_params(lines)
    func_attrs = find_all_func_attr(lines)
    func_vgprs = find_all_vgpr_usage(asm)
    func_sgprs = find_all_sgpr_usage(asm)

    # print(func_sgprs)
    # print(func_attrs)
    launch_params = {}
    for kernel in schedule["kernels"]:
        launch_params[kernel["name"]] = kernel["launch_params"]

    lines = replace_global_with_device(lines)
    # lines = replace_blockIdx_with_task_idx(lines)
    # lines = replace_threadIdx_with_thread_idx(lines)
    # lines = replace_shared_memory(lines)
    # lines = add_device_function_param(lines)
    lines = add_global_definition(lines)
    lines.extend(generate_device_wrapper(func_params, func_attrs, launch_params, func_vgprs, func_sgprs))
    lines.extend(generate_global_caller(func_params))
    # lines.extend(generate_persistent_thread(func_params, func_attrs, launch_params))
    # lines.extend(generate_global_wrappers(func_params))
    # lines.extend(generate_merged_kernel(func_params))
    # lines.extend(generate_extern_kernel(func_params))
    # lines.extend(generate_get_address_kernel(func_params))
    lines.extend(generate_dim_transformations(launch_params))
    lines.extend(generate_call_framework())
    lines.extend(generate_merge_framework())
    output_f_name = sys.argv[1][:sys.argv[1].rfind(".")] + ".trans.cu"

    f = open(output_f_name, "w")
    f.writelines(lines)
    f.close()
    