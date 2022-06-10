# NVCC= nvcc

# INCLUDES=-I/usr/local/cuda/include -I/opt/rocm/hip/include

# LIBS=-lcuda

# ARCH=70

CC=hipcc

INCLUDES= -I/opt/rocm/hip/include

SUBDIRS = ${shell ls -d */ | sed 's/\///' -}


CURRENT_PATH = ${shell pwd}

CURRENT_DIR = ${shell basename $(CURRENT_PATH)}

ARCH=gfx906

all: 
	@for dir in $(SUBDIRS); do \
		make -C $$dir -f ../Makefile build_subdir;\
	done

transform:
	@for dir in $(SUBDIRS); do \
	    make -C $$dir -f ../Makefile transform_subdir;\
	done

profile:
	for dir in $(SUBDIRS); do \
		echo "profiling $$dir"; \
	    profiler $$dir;\
	done

build_subdir: ${CURRENT_DIR}.raw.co ${CURRENT_DIR}.trans.co ${CURRENT_DIR}.be.co 

transform_subdir: ${CURRENT_DIR}-hip-amdgcn-amd-amdhsa-gfx906.s
	python3 ../../script/transform_kernel.py ${CURRENT_DIR}.cu ${CURRENT_DIR}.json ${CURRENT_DIR}-hip-amdgcn-amd-amdhsa-gfx906.s
	python3 ../../script/best_effort_kernel.py ${CURRENT_DIR}.cu



%.be.cu: %.cu
	python3 ../../script/best_effort_kernel.py ${CURRENT_DIR}.cu

%.preempt.cu: %.cu
	python3 ../../script/preemptable_kernel.py ${CURRENT_DIR}.cu ${CURRENT_DIR}.json	

%.trans.cu: %.cu ${CURRENT_DIR}-hip-amdgcn-amd-amdhsa-gfx906.s
	python3 ../../script/transform_kernel.py ${CURRENT_DIR}.cu ${CURRENT_DIR}.json ${CURRENT_DIR}-hip-amdgcn-amd-amdhsa-gfx906.s

%-hip-amdgcn-amd-amdhsa-gfx906.s: %.cu
	$(CC) $(INCLUDES) --save-temps --genco --offload-arch=${ARCH} $< -o $*.raw.co
	rm *.bc *.cui *.o *.out *.txt

%.raw.co: %.cu
	$(CC) $(INCLUDES) --save-temps --genco --offload-arch=${ARCH} $< -o $@
	rm *.bc *.cui *.o *.out *.txt

%.be.co: %.be.cu
	$(CC) $(INCLUDES) --genco --offload-arch=${ARCH} $< -o $@

%.trans.co: %.trans.cu
	@"/opt/rocm/llvm/bin/clang" -cc1 -mllvm --amdhsa-code-object-version=4 -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-linux-gnu -E -save-temps=cwd -disable-free -disable-llvm-verifier -discard-value-names -main-file-name $< -mrelocation-model pic -pic-level 1 -fhalf-no-semantic-interposition -mframe-pointer=none -fno-rounding-math -aux-target-cpu x86-64 -fcuda-is-device -mllvm -amdgpu-internalize-symbols -fcuda-allow-variadic-functions -fvisibility hidden -fapply-global-visibility-to-externs -mlink-builtin-bitcode /opt/rocm-4.3.0/amdgcn/bitcode/hip.bc -mlink-builtin-bitcode /opt/rocm-4.3.0/amdgcn/bitcode/ocml.bc -mlink-builtin-bitcode /opt/rocm-4.3.0/amdgcn/bitcode/ockl.bc -mlink-builtin-bitcode /opt/rocm-4.3.0/amdgcn/bitcode/oclc_daz_opt_off.bc -mlink-builtin-bitcode /opt/rocm-4.3.0/amdgcn/bitcode/oclc_unsafe_math_off.bc -mlink-builtin-bitcode /opt/rocm-4.3.0/amdgcn/bitcode/oclc_finite_only_off.bc -mlink-builtin-bitcode /opt/rocm-4.3.0/amdgcn/bitcode/oclc_correctly_rounded_sqrt_on.bc -mlink-builtin-bitcode /opt/rocm-4.3.0/amdgcn/bitcode/oclc_wavefrontsize64_on.bc -mlink-builtin-bitcode /opt/rocm-4.3.0/amdgcn/bitcode/oclc_isa_version_906.bc -target-cpu gfx906 -fno-split-dwarf-inlining -debugger-tuning=gdb -v -resource-dir /opt/rocm-4.3.0/llvm/lib/clang/13.0.0 -internal-isystem /opt/rocm-4.3.0/llvm/lib/clang/13.0.0/include/cuda_wrappers -internal-isystem /opt/rocm-4.3.0/include -include __clang_hip_runtime_wrapper.h -isystem /opt/rocm-4.3.0/llvm/lib/clang/13.0.0/include/.. -isystem /opt/rocm-4.3.0/hsa/include -isystem /opt/rocm-4.3.0/hip/include -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/x86_64-linux-gnu/c++/7.5.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/x86_64-linux-gnu/c++/7.5.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0/backward -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/x86_64-linux-gnu/c++/7.5.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/x86_64-linux-gnu/c++/7.5.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0/backward -internal-isystem /usr/local/include -internal-isystem /opt/rocm-4.3.0/llvm/lib/clang/13.0.0/include -internal-externc-isystem /usr/include/x86_64-linux-gnu -internal-externc-isystem /include -internal-externc-isystem /usr/include -internal-isystem /usr/local/include -internal-isystem /opt/rocm-4.3.0/llvm/lib/clang/13.0.0/include -internal-externc-isystem /usr/include/x86_64-linux-gnu -internal-externc-isystem /include -internal-externc-isystem /usr/include -O3 -std=c++11 -fdeprecated-macro -fno-autolink  -ferror-limit 19 -fhip-new-launch-api -fgnuc-version=4.2.1 -fcxx-exceptions -fexceptions -fcolor-diagnostics -vectorize-loops -vectorize-slp -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false -cuid=8a385692c0a935c2 -fcuda-allow-variadic-functions -munsafe-fp-atomics -faddrsig -o $*.trans-hip-amdgcn-amd-amdhsa-gfx906.cui -x hip $<
	@"/opt/rocm/llvm/bin/clang" -cc1 -mllvm --amdhsa-code-object-version=4 -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-linux-gnu -emit-llvm-bc -emit-llvm-uselists -save-temps=cwd -disable-free -disable-llvm-verifier -discard-value-names -main-file-name $< -mrelocation-model pic -pic-level 1 -fhalf-no-semantic-interposition -mframe-pointer=none -fno-rounding-math -aux-target-cpu x86-64 -fcuda-is-device -mllvm -amdgpu-internalize-symbols -fcuda-allow-variadic-functions -fvisibility hidden -fapply-global-visibility-to-externs -mlink-builtin-bitcode /opt/rocm-4.3.0/amdgcn/bitcode/hip.bc -mlink-builtin-bitcode /opt/rocm-4.3.0/amdgcn/bitcode/ocml.bc -mlink-builtin-bitcode /opt/rocm-4.3.0/amdgcn/bitcode/ockl.bc -mlink-builtin-bitcode /opt/rocm-4.3.0/amdgcn/bitcode/oclc_daz_opt_off.bc -mlink-builtin-bitcode /opt/rocm-4.3.0/amdgcn/bitcode/oclc_unsafe_math_off.bc -mlink-builtin-bitcode /opt/rocm-4.3.0/amdgcn/bitcode/oclc_finite_only_off.bc -mlink-builtin-bitcode /opt/rocm-4.3.0/amdgcn/bitcode/oclc_correctly_rounded_sqrt_on.bc -mlink-builtin-bitcode /opt/rocm-4.3.0/amdgcn/bitcode/oclc_wavefrontsize64_on.bc -mlink-builtin-bitcode /opt/rocm-4.3.0/amdgcn/bitcode/oclc_isa_version_906.bc -target-cpu gfx906 -fno-split-dwarf-inlining -debugger-tuning=gdb -v -resource-dir /opt/rocm-4.3.0/llvm/lib/clang/13.0.0 -O3 -std=c++11 -fdeprecated-macro -fno-autolink  -ferror-limit 19 -fhip-new-launch-api -fgnuc-version=4.2.1 -fcxx-exceptions -fexceptions -fcolor-diagnostics -vectorize-loops -vectorize-slp -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false -disable-llvm-passes -cuid=8a385692c0a935c2 -fcuda-allow-variadic-functions -munsafe-fp-atomics -faddrsig -o $*.trans-hip-amdgcn-amd-amdhsa-gfx906.bc -x hip-cpp-output $*.trans-hip-amdgcn-amd-amdhsa-gfx906.cui
	@"/opt/rocm/llvm/bin/clang" -cc1 -mllvm --amdhsa-code-object-version=4 -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-linux-gnu -S -save-temps=cwd -disable-free -disable-llvm-verifier -discard-value-names -main-file-name $< -mrelocation-model pic -pic-level 1 -fhalf-no-semantic-interposition -mframe-pointer=none -fno-rounding-math -aux-target-cpu x86-64 -fcuda-is-device -mllvm -amdgpu-internalize-symbols -fcuda-allow-variadic-functions -fvisibility hidden -fapply-global-visibility-to-externs -mlink-builtin-bitcode /opt/rocm-4.3.0/amdgcn/bitcode/hip.bc -mlink-builtin-bitcode /opt/rocm-4.3.0/amdgcn/bitcode/ocml.bc -mlink-builtin-bitcode /opt/rocm-4.3.0/amdgcn/bitcode/ockl.bc -mlink-builtin-bitcode /opt/rocm-4.3.0/amdgcn/bitcode/oclc_daz_opt_off.bc -mlink-builtin-bitcode /opt/rocm-4.3.0/amdgcn/bitcode/oclc_unsafe_math_off.bc -mlink-builtin-bitcode /opt/rocm-4.3.0/amdgcn/bitcode/oclc_finite_only_off.bc -mlink-builtin-bitcode /opt/rocm-4.3.0/amdgcn/bitcode/oclc_correctly_rounded_sqrt_on.bc -mlink-builtin-bitcode /opt/rocm-4.3.0/amdgcn/bitcode/oclc_wavefrontsize64_on.bc -mlink-builtin-bitcode /opt/rocm-4.3.0/amdgcn/bitcode/oclc_isa_version_906.bc -target-cpu gfx906 -fno-split-dwarf-inlining -debugger-tuning=gdb -v -resource-dir /opt/rocm-4.3.0/llvm/lib/clang/13.0.0 -O3 -std=c++11 -fno-autolink -ferror-limit 19 -fhip-new-launch-api -fgnuc-version=4.2.1 -fcolor-diagnostics -vectorize-loops -vectorize-slp -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false -cuid=8a385692c0a935c2 -fcuda-allow-variadic-functions -munsafe-fp-atomics -faddrsig -o $*.trans-hip-amdgcn-amd-amdhsa-gfx906.s -x ir $*.trans-hip-amdgcn-amd-amdhsa-gfx906.bc
	@python3 ../../script/replace_register_usage.py $*.trans-hip-amdgcn-amd-amdhsa-gfx906.s
	python3 ../../script/generate_asm_loop.py $*.trans-hip-amdgcn-amd-amdhsa-gfx906.s $*.json
	@"/opt/rocm/llvm/bin/clang" -cc1as -mllvm --amdhsa-code-object-version=4 -triple amdgcn-amd-amdhsa -filetype obj -main-file-name $< -target-cpu gfx906 -dwarf-version=4 -mrelocation-model pic --mrelax-relocations -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false -o $*.trans-hip-amdgcn-amd-amdhsa-gfx906.o $*.trans-hip-amdgcn-amd-amdhsa-gfx906.s
	@"/opt/rocm/llvm/bin/lld" -flavor gnu --no-undefined -shared -plugin-opt=-amdgpu-internalize-symbols -plugin-opt=mcpu=gfx906 -plugin-opt=O3 -plugin-opt=-amdgpu-early-inline-all=true -plugin-opt=-amdgpu-function-calls=false -save-temps -o $*.trans-hip-amdgcn-amd-amdhsa-gfx906.out $*.trans-hip-amdgcn-amd-amdhsa-gfx906.o
	@"/opt/rocm/llvm/bin/clang-offload-bundler" -type=o -bundle-align=4096 -targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--gfx906 -inputs=/dev/null,$*.trans-hip-amdgcn-amd-amdhsa-gfx906.out -outputs=$@
	@echo build $@
	rm *.bc *.cui *.o *.out *.txt
	
# @"/opt/rocm/llvm/bin/clang-13" -cc1 -mllvm --amdhsa-code-object-version=4 -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-linux-gnu -E -save-temps=cwd -disable-free -disable-llvm-verifier -discard-value-names -main-file-name $< -mrelocation-model pic -pic-level 1 -mframe-pointer=none -fno-rounding-math -mconstructor-aliases -aux-target-cpu x86-64 -fcuda-is-device -mllvm -amdgpu-internalize-symbols -fcuda-allow-variadic-functions -fvisibility hidden -fapply-global-visibility-to-externs -mlink-builtin-bitcode /opt/rocm/amdgcn/bitcode/hip.bc -mlink-builtin-bitcode /opt/rocm/amdgcn/bitcode/ocml.bc -mlink-builtin-bitcode /opt/rocm/amdgcn/bitcode/ockl.bc -mlink-builtin-bitcode /opt/rocm/amdgcn/bitcode/oclc_daz_opt_off.bc -mlink-builtin-bitcode /opt/rocm/amdgcn/bitcode/oclc_unsafe_math_off.bc -mlink-builtin-bitcode /opt/rocm/amdgcn/bitcode/oclc_finite_only_off.bc -mlink-builtin-bitcode /opt/rocm/amdgcn/bitcode/oclc_correctly_rounded_sqrt_on.bc -mlink-builtin-bitcode /opt/rocm/amdgcn/bitcode/oclc_wavefrontsize64_on.bc -mlink-builtin-bitcode /opt/rocm/amdgcn/bitcode/oclc_isa_version_906.bc -target-cpu gfx906 -fno-split-dwarf-inlining -debugger-tuning=gdb -resource-dir /opt/rocm/llvm/lib/clang/12.0.0 -internal-isystem /opt/rocm/llvm/lib/clang/12.0.0/include/cuda_wrappers -internal-isystem /opt/rocm/include -include __clang_hip_runtime_wrapper.h -isystem /opt/rocm/llvm/lib/clang/12.0.0/include/.. -isystem /opt/rocm/hsa/include -isystem /opt/rocm/hip/include -D __HIP_ROCclr__ -D __HIP_ROCclr__ -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/x86_64-linux-gnu/c++/7.5.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/x86_64-linux-gnu/c++/7.5.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0/backward -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/x86_64-linux-gnu/c++/7.5.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/x86_64-linux-gnu/c++/7.5.0 -internal-isystem /usr/lib/gcc/x86_64-linux-gnu/7.5.0/../../../../include/c++/7.5.0/backward -internal-isystem /usr/local/include -internal-isystem /opt/rocm/llvm/lib/clang/12.0.0/include -internal-externc-isystem /usr/include/x86_64-linux-gnu -internal-externc-isystem /include -internal-externc-isystem /usr/include -internal-isystem /usr/local/include -internal-isystem /opt/rocm/llvm/lib/clang/12.0.0/include -internal-externc-isystem /usr/include/x86_64-linux-gnu -internal-externc-isystem /include -internal-externc-isystem /usr/include -O3 -std=c++11 -fdeprecated-macro -fno-autolink -ferror-limit 19 -fhip-new-launch-api -fgnuc-version=4.2.1 -fcxx-exceptions -fexceptions -fcolor-diagnostics -vectorize-loops -vectorize-slp -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false -fcuda-allow-variadic-functions -faddrsig -o $*.trans-hip-amdgcn-amd-amdhsa-gfx906.cui -x hip $<
# @"/opt/rocm/llvm/bin/clang-13" -cc1 -mllvm --amdhsa-code-object-version=4 -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-linux-gnu -emit-llvm-bc -emit-llvm-uselists -save-temps=cwd -disable-free -disable-llvm-verifier -discard-value-names -main-file-name $< -mrelocation-model pic -pic-level 1 -mframe-pointer=none -fno-rounding-math -mconstructor-aliases -aux-target-cpu x86-64 -fcuda-is-device -mllvm -amdgpu-internalize-symbols -fcuda-allow-variadic-functions -fvisibility hidden -fapply-global-visibility-to-externs -mlink-builtin-bitcode /opt/rocm/amdgcn/bitcode/hip.bc -mlink-builtin-bitcode /opt/rocm/amdgcn/bitcode/ocml.bc -mlink-builtin-bitcode /opt/rocm/amdgcn/bitcode/ockl.bc -mlink-builtin-bitcode /opt/rocm/amdgcn/bitcode/oclc_daz_opt_off.bc -mlink-builtin-bitcode /opt/rocm/amdgcn/bitcode/oclc_unsafe_math_off.bc -mlink-builtin-bitcode /opt/rocm/amdgcn/bitcode/oclc_finite_only_off.bc -mlink-builtin-bitcode /opt/rocm/amdgcn/bitcode/oclc_correctly_rounded_sqrt_on.bc -mlink-builtin-bitcode /opt/rocm/amdgcn/bitcode/oclc_wavefrontsize64_on.bc -mlink-builtin-bitcode /opt/rocm/amdgcn/bitcode/oclc_isa_version_906.bc -target-cpu gfx906 -fno-split-dwarf-inlining -debugger-tuning=gdb -resource-dir /opt/rocm/llvm/lib/clang/12.0.0 -O3 -std=c++11 -fdeprecated-macro -fno-autolink -ferror-limit 19 -fhip-new-launch-api -fgnuc-version=4.2.1 -fcxx-exceptions -fexceptions -fcolor-diagnostics -vectorize-loops -vectorize-slp -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false -disable-llvm-passes -fcuda-allow-variadic-functions -faddrsig -o $*.trans-hip-amdgcn-amd-amdhsa-gfx906.bc -x hip-cpp-output $*.trans-hip-amdgcn-amd-amdhsa-gfx906.cui
# @"/opt/rocm/llvm/bin/clang-13" -cc1 -mllvm --amdhsa-code-object-version=4 -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-linux-gnu -S -save-temps=cwd -disable-free -disable-llvm-verifier -discard-value-names -main-file-name $< -mrelocation-model pic -pic-level 1 -mframe-pointer=none -fno-rounding-math -mconstructor-aliases -aux-target-cpu x86-64 -fcuda-is-device -mllvm -amdgpu-internalize-symbols -fcuda-allow-variadic-functions -fvisibility hidden -fapply-global-visibility-to-externs -mlink-builtin-bitcode /opt/rocm/amdgcn/bitcode/hip.bc -mlink-builtin-bitcode /opt/rocm/amdgcn/bitcode/ocml.bc -mlink-builtin-bitcode /opt/rocm/amdgcn/bitcode/ockl.bc -mlink-builtin-bitcode /opt/rocm/amdgcn/bitcode/oclc_daz_opt_off.bc -mlink-builtin-bitcode /opt/rocm/amdgcn/bitcode/oclc_unsafe_math_off.bc -mlink-builtin-bitcode /opt/rocm/amdgcn/bitcode/oclc_finite_only_off.bc -mlink-builtin-bitcode /opt/rocm/amdgcn/bitcode/oclc_correctly_rounded_sqrt_on.bc -mlink-builtin-bitcode /opt/rocm/amdgcn/bitcode/oclc_wavefrontsize64_on.bc -mlink-builtin-bitcode /opt/rocm/amdgcn/bitcode/oclc_isa_version_906.bc -target-cpu gfx906 -fno-split-dwarf-inlining -debugger-tuning=gdb -resource-dir /opt/rocm/llvm/lib/clang/12.0.0 -O3 -std=c++11 -fno-autolink -ferror-limit 19 -fhip-new-launch-api -fgnuc-version=4.2.1 -fcolor-diagnostics -vectorize-loops -vectorize-slp -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false -fcuda-allow-variadic-functions -faddrsig -o $*.trans-hip-amdgcn-amd-amdhsa-gfx906.s -x ir $*.trans-hip-amdgcn-amd-amdhsa-gfx906.bc
# @python3 ../../script/replace_register_usage.py $*.trans-hip-amdgcn-amd-amdhsa-gfx906.s
# python3 ../../script/generate_asm_loop.py $*.trans-hip-amdgcn-amd-amdhsa-gfx906.s $*.json
# @"/opt/rocm/llvm/bin/clang-13" -cc1as -mllvm --amdhsa-code-object-version=4 -triple amdgcn-amd-amdhsa -filetype obj -main-file-name $< -target-cpu gfx906 -dwarf-version=4 -mrelocation-model pic --mrelax-relocations -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false -o $*.trans-hip-amdgcn-amd-amdhsa-gfx906.o $*.trans-hip-amdgcn-amd-amdhsa-gfx906.s
# @"/opt/rocm/llvm/bin/lld" -flavor gnu --no-undefined -shared -plugin-opt=-amdgpu-internalize-symbols -plugin-opt=mcpu=gfx906 -plugin-opt=O3 -plugin-opt=-amdgpu-early-inline-all=true -plugin-opt=-amdgpu-function-calls=false -save-temps -o $*.trans-hip-amdgcn-amd-amdhsa-gfx906.out $*.trans-hip-amdgcn-amd-amdhsa-gfx906.o
# @"/opt/rocm/llvm/bin/clang-offload-bundler" -type=o -bundle-align=4096 -targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--gfx906 -inputs=/dev/null,$*.trans-hip-amdgcn-amd-amdhsa-gfx906.out -outputs=$@
# @echo build $@
# rm *.bc *.cui *.o *.out *.txt

%.asm: %-hip-amdgcn-amd-amdhsa-gfx906.s
	"/opt/rocm/llvm/bin/clang" -cc1as -mllvm --amdhsa-code-object-version=4 -triple amdgcn-amd-amdhsa -filetype obj -main-file-name $< -target-cpu gfx906 -dwarf-version=4 -mrelocation-model pic --mrelax-relocations -mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false -o $*.trans-hip-amdgcn-amd-amdhsa-gfx906.o $*.trans-hip-amdgcn-amd-amdhsa-gfx906.s
	"/opt/rocm/llvm/bin/lld" -flavor gnu --no-undefined -shared -plugin-opt=-amdgpu-internalize-symbols -plugin-opt=mcpu=gfx906 -plugin-opt=O3 -plugin-opt=-amdgpu-early-inline-all=true -plugin-opt=-amdgpu-function-calls=false -save-temps -o $*.trans-hip-amdgcn-amd-amdhsa-gfx906.out $*.trans-hip-amdgcn-amd-amdhsa-gfx906.o
	"/opt/rocm/llvm/bin/clang-offload-bundler" -type=o -bundle-align=4096 -targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--gfx906 -inputs=/dev/null,$*.trans-hip-amdgcn-amd-amdhsa-gfx906.out -outputs=$*.trans.co

clean_temp:
	rm -f */*.s */*.out */*.o */*.bc */*.txt */*.cui	

clean:
	rm -f */*.co */*.bc */*.cui */*.o */*.out */*.txt */*.s */*.pointer.cu */*.prtopt.cu */*.ptrraw.cu */*.preempt.cu