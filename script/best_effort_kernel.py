import sys, json
from transform_kernel import replace_global_with_device,replace_blockIdx_with_task_idx, find_all_func_params
from transform_kernel import add_device_function_param,add_global_definition, generate_function_declaration


def generate_global_wrappers(func_params):
    result = []
    for func_name, func_param in func_params.items():
        params = [
            {"type": "int*", "name": "preempted"},
            {"type": "int*", "name": "task_slot"},
        ]
        params.extend(func_param)
        params_name = []
        params_type_name = []
        for param in func_param:
            params_name.append(param["name"])
        for param in params:
            params_type_name.append(param["type"] + " " + param["name"])
        params_def = ", ".join(params_type_name)
        params_call = ", ".join(params_name)
        func_template = """
extern "C" __global__ void {func_name}({params_def}) {{
    if (*preempted) return;
    {func_name}_device({params_call});
    if (threadIdx.x + threadIdx.y + threadIdx.z == 0)
        atomicAdd(task_slot, 1);
}}        
""".format(
    func_name = func_name, 
    params_def = params_def,
    params_call = params_call
)       
        result.extend(func_template.splitlines(True))
    return result

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python best_effort_kernel.py input_file.cu")
        exit(0)

    f = open(sys.argv[1], "r")
    lines = f.readlines()
    f.close()


    func_params = find_all_func_params(lines)
    lines = replace_global_with_device(lines)
    lines.extend(generate_global_wrappers(func_params))

    output_f_name = sys.argv[1][:sys.argv[1].rfind(".")] + ".be.cu"

    f = open(output_f_name, "w")
    f.writelines(lines)
    f.close()
    