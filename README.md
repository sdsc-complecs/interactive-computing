# COMPLECS Interactive Computing Exercises

These exercises use sample code available from these two repositories:
* 

## [Table of Contents](#top)
* [Accessing Interactive Compute Nodes on Expanse](#int-nodes)
  * [Accessing Interactive Compute Nodes on Expanse: CPU](#int-nodes-cpu)
  * [Accessing Interactive Compute Nodes on Expanse: GPU](#int-nodes-cpu)
* [Launching Notebooks with Galyleo](#galy-notebk)
  * [Launching Notebooks with Galyleo: CPU](#galy-notebk-cpu)
  * [Launching Notebooks with Galyleo: GPU](#galy-notebk-gpu)
* [Using X11 to Launch GUI on Expanse](#exp-x11)
* [Using the Expanse Portal](#exp-portal)
  * [Launching Jupyter Notebook](#exp-portal-jupnb)

## Using X11 to Launch GUI on Expanse] <a id="exp-x11"></a>
* In this example, we'll use X11 Forwarding to Expanse Interactive Node to run Matlab
* Note: MacOS/Ventura has an X11 problem that is being debugged.
* You will launch 2 connections from your expanse login account
  * Connection #1: request the interactive node
  * Connection #2: connect to the interactive node using X11 ssh connection

* Setup Connection #1 to an interactive node:
  * Connect to Expanse from local computing using the "-Y" for X11 connection

```
[localhost ]$ ssh -Y mthomas@login.expanse.sdsc.edu
[mthomas@login02 ~]$ srun --partition=debug  --pty --account=use300 --nodes=1 --ntasks-per-node=4  --mem=8G -t 00:30:00 --wait=0 --export=ALL /bin/bash
srun: job 29969059 queued and waiting for resources
srun: job 29969059 has been allocated resources
[mthomas@exp-9-55 ~]$ 
```

* Connection #2: connect to the interactive node with X11 flags on; set up module environment and launch app.
  * Connect to Expanse from local computing using the "-Y" for X11 connection
  * Then connect to the interactive node
```
[localhost ]$ ssh -Y mthomas@login.expanse.sdsc.edu
[mthomas@login01 ~] ssh -Y exp-9-55
Last login: Wed Dec  6 19:21:56 2023 from 10.21.0.19
[mthomas@exp-9-55 ~]$ module load  gpu/0.17.3b
[mthomas@exp-9-55 ~]$ module load matlab/2022b/nmbr5dd
[mthomas@exp-9-55 ~]$ matlab
```
  
[Back to Top](#top)

## Accessing Interactive Compute Nodes on Expanse <a id="int-nodes"></a>
* In this exercise, you will learn to launch interactive sessions on a compute node from the command line.

### Accessing Interactive Compute Nodes on Expanse: CPU <a id="int-nodes-cpu"></a>
* access CPU node
  
```
[mthomas@login02 ~]$ srun --partition=debug  --pty --account=<<project>> --nodes=1 --ntasks-per-node=4 \
    --mem=8G -t 00:30:00 --wait=0 --export=ALL /bin/bash
> ^C
[mthomas@login02 ~]$ srun --partition=debug  --pty --account=use300 --nodes=1 --ntasks-per-node=4     --mem=8G -t 00:30:00 --wait=0 --export=ALL /bin/bash
srun: job 29968362 queued and waiting for resources
. . .
srun: job 29968362 has been allocated resources
[mthomas@exp-9-56 ~]$ 
[mthomas@exp-9-56 ~]$ hostname
exp-9-56
[mthomas@exp-9-56 ~]$ lscpu 
Architecture:        x86_64
CPU op-mode(s):      32-bit, 64-bit
Byte Order:          Little Endian
CPU(s):              128
On-line CPU(s) list: 0-127
Thread(s) per core:  1
Core(s) per socket:  64
Socket(s):           2
NUMA node(s):        8
Vendor ID:           AuthenticAMD
CPU family:          23
Model:               49
Model name:          AMD EPYC 7742 64-Core Processor
Stepping:            0
CPU MHz:             3389.388
BogoMIPS:            4491.83
Virtualization:      AMD-V
L1d cache:           32K
L1i cache:           32K
L2 cache:            512K
L3 cache:            16384K
NUMA node0 CPU(s):   0-15
NUMA node1 CPU(s):   16-31
NUMA node2 CPU(s):   32-47
NUMA node3 CPU(s):   48-63
NUMA node4 CPU(s):   64-79
NUMA node5 CPU(s):   80-95
NUMA node6 CPU(s):   96-111
NUMA node7 CPU(s):   112-127
Flags:               fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl nonstop_tsc cpuid extd_apicid aperfmperf pni pclmulqdq monitor ssse3 fma cx16 sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 hw_pstate ssbd mba ibrs ibpb stibp vmmcall fsgsbase bmi1 avx2 smep bmi2 cqm rdt_a rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local clzero irperf xsaveerptr wbnoinvd amd_ppin arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold avic v_vmsave_vmload vgif v_spec_ctrl umip rdpid overflow_recov succor smca sme sev sev_es

```

* Next, try to compile and run the mpi-prim code on the interactive node
* once you have the node, cd to the mpi-prime directory:

```
[mthomas@exp-9-55 ~]$ cd /home/mthomas/hpctr-examples/calc-prime
[mthomas@exp-9-55 calc-prime]$ ll
total 72
-rwxr-xr-x 1 mthomas use300 22880 Apr 17 21:00 mpi_prime
-rw-r--r-- 1 mthomas use300  1157 Oct 11  2023 mpi_prime.25652153.exp-9-55.out
-rw-r--r-- 1 mthomas use300  1082 Oct 11  2023 mpi_prime.25659610.exp-9-56.out
-rw-r--r-- 1 mthomas use300  1157 Oct 11  2023 mpi_prime.25659611.exp-9-56.out
-rw-r--r-- 1 mthomas use300  5194 Oct 11  2023 mpi_prime.c
-rw-r--r-- 1 mthomas use300   874 Oct 11  2023 mpi-prime-slurm.sb
-rw-r--r-- 1 mthomas use300  1441 Oct 11  2023 README.txt
[mthomas@exp-9-55 calc-prime]$ cat README.txt 
[1] Compile:

module purge 
module load slurm
module load cpu
module load gcc/10.2.0
module load openmpi/4.1.1

mpicc -o mpi_prime mpi_prime.c 


[2] Run as batch job:

    sbatch mpi-prime-slurm.sb

To pass value to script:
    sbatch --export=NHI=250000 mpi-prime-slurm.sb 

NOTE: for other compilers, replace "gcc"
with the one you want to use.
 
[3] Run using an interactive node:
Method [3a]
Request the interactive node using the "srun" command:

srun --partition=debug  --pty --account=use300 --nodes=1 --ntasks-per-node=24  --mem=8G -t 00:30:00 --wait=0 --export=ALL /bin/bash

Run the code using mpirun:
mpirun -n 64 ./mpi_prime 5000000
```
* Run the calculation using option [3a]
```
[mthomas@exp-9-55 calc-prime]$ which mpirun
/cm/shared/apps/spack/0.17.3/cpu/b/opt/spack/linux-rocky8-zen2/gcc-10.2.0/openmpi-4.1.1-ygduf2ryo2scwdtpl4wftbmlz2xubbrv/bin/mpirun
[mthomas@exp-9-55 calc-prime]$ mpirun -n 4 ./mpi_prime 10000
The argument supplied is 10000
17 April 2024 09:02:35 PM

PRIME_MPI
 n_hi= 10000
  C/MPI version

  An MPI example program to count the number of primes.
  The number of processes is 4

         N        Pi          Time

         1         0        0.000136
         2         1        0.000001
         4         2        0.000001
         8         4        0.000036
        16         6        0.000001
        32        11        0.000039
        64        18        0.000002
       128        31        0.000005
       256        54        0.000034
       512        97        0.000069
      1024       172        0.000192
      2048       309        0.000666
      4096       564        0.002389
      8192      1028        0.008458

PRIME_MPI - Master process:
  Normal end of execution.

17 April 2024 09:02:35 PM


```

[Back to Top](#top)

### Accessing Interactive Compute Nodes on Expanse: GPU <a id="int-nodes-Gpu"></a>
* The following example  requests a GPU node, 10 cores, 1 GPU and 96G  in the debug partition for 30 minutes.
* To ensure the GPU environment is properly loaded, please be sure run both the module purge and module restore commands.

```
[mthomas@login02 ~]$ srun --partition=gpu-debug --pty --account=use300 --ntasks-per-node=10 --nodes=1 --mem=96G --gpus=1 -t 00:30:00 --wait=0 --export=ALL /bin/bash
srun: job 29968716 queued and waiting for resources
srun: job 29968716 has been allocated resources
[mthomas@exp-7-59 ~]$ hostname
exp-7-59
[mthomas@exp-7-59 ~]$ nvidia-smi
Wed Apr 17 20:46:35 2024       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.85.12    Driver Version: 525.85.12    CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  On   | 00000000:18:00.0 Off |                    0 |
| N/A   34C    P0    41W / 300W |      0MiB / 32768MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
|        ID   ID                                                   Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
[mthomas@exp-7-59 ~]$ exit
exit
[mthomas@login02 ~]$ 
```

* Next, navigate to the ```hpc-training``` directory, and try to compile and run the cuda based hello world code:
  
```
[mthomas@exp-7-59 ~]$cd ~/hpctr-examples/cuda/hello-world
[mthomas@exp-7-59 hello-world]$ cat README.txt 
Hello World (GPU/CUDA)
------------------------------------------------------------------------
Updated by Mary Thomas (mthomas at ucsd.edu)
August, 2023
------------------------------------------------------------------------

[1] Load the correct modules for the CUDA Compiler:

module purge
module load gpu/0.15.4
module load gcc/7.2.0
module load cuda/11.0.2
module load slurm

[2] compile:

nvcc -o hello_world hello_world.cu

[3] run from interactive node

./hello_world
[mthomas@exp-7-59 hello-world]$ module purge
module load gpu/0.15.4
module load gcc/7.2.0
module load cuda/11.0.2
module load slurm
[mthomas@exp-7-59 hello-world]$ nvcc -o hello_world hello_world.cu
[mthomas@exp-7-59 hello-world]$ ./hello_world 
Hello,  SDSC HPC Training World!
[mthomas@exp-7-59 hello-world]$
```
[Back to Top](#top)

## [Launching Notebooks with Galyleo] <a id="galy-notebk"></a>
Launch Jupyter notebooks on Expanse CPUs and GPUs using the Galyleo shell utility:  https://github.com/mkandes/galyleo 

### Launching Notebooks with Galyleo: CPU <a id="galy-notebk-cpu"></a>

```
export PATH="/cm/shared/apps/sdsc/galyleo:${PATH}"
[username@login01 ~]$ which galyleo
```

[Back to Top](#top)

### Launching Notebooks with Galyleo: GPU <a id="galy-notebk-gpu"></a>
* Follow examples on the slide titled _Launching GPU notebooks using galyleo_
* Run the Hello World example in the notebooks folder.
```
export PATH="/cm/shared/apps/sdsc/galyleo:${PATH}"
[username@login01 ~]$ which galyleo
[username@login01 ~]$ galyleo launch --account use300 --partition gpu-debug --cpus 1 --memory 93 --gpus 1 --time-limit 00:5:00 --env-modules singularitypro --sif /cm/shared/apps/containers/singularity/pytorch/pytorch-latest.sif --bind /expanse,/scratch --nv 
[snip]
Submitted Jupyter launch script to Slurm. Your SLURM_JOB_ID is 9773912
[snip]
Your Jupyter notebook session will begin once compute resources are allocated to your job by the scheduler.
https://grief-fantastic-given.expanse-user-content.sdsc.edu?token=5097acb6f32ab82dd51b4524c267d2fd
[username@login01 ~]$ squeue -u username
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON) 
           9773912 gpu-debug galyleo-  username PD       0:00      1 (None) 
[username@login01 ~]$ squeue -u username
             JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON) 
           9773912 gpu-debug galyleo-  username  R       0:20      1 exp-7-59 
```
* Monitor URL until notebook launches. You can then run the Hello World example:
```
/notebook-examples/Hello_World/

```

## Using the Expanse Portal <a id="exp-portal"></a>
* Log onto the Expanse portal:  https://portal.expanse.sdsc.edu
* Practice navigating your folders and directories
* Launch the main applications:
  * Matlab
  * Jupyter Notebook
 
### Launching Jupyter Notebook <a id="exp-portal-jupnb"></a>
``` 
Account: use300
Partition: (Please choose the gpu, gpu-shared, or gpu-preempt as the partition if using gpus): debug
Time limit (min): 30
Number of cores: 1
Memory required per node (GB): 2
GPUs (optional): 0
Singularity Image File Location: (Use your own or to include from existing container library at /cm/shared/apps/container e.g., /cm/shared/apps/containers/singularity/pytorch/pytorch-latest.sif)
/cm/shared/apps/containers/singularity/pytorch/pytorch-latest.sif
Environment modules to be loaded (E.g., to use latest version of system Anaconda3 include cpu,gcc,anaconda3):   singularitypro
Conda Environment (Enter your own conda environment if any):
Conda Init (Provide path to conda initialization scripts):
Conda Yaml (Upload a yaml file to build the conda environment at runtime) No file chosen:
Turn on use of mamba for speeding up conda-yml installs: 
Enable use of new caching mechanism that will store and reuse conda-yml created environments using conda-pack !????
Reservation: 
QoS:
Working directory:  HOME
Type:  JupyterLab
```

[Back to Top](#top)
