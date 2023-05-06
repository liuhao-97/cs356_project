Notice the project is based on Vitis-AI2.5 with U55C.

First the evironment need to be settled.

For flash the FPGA, follow the requirement of Vitis-AI.

```
sudo apt install ./xilinx-u55c-gen3x16-xdma-base_2-3349938_all.deb
sudo /opt/xilinx/xrt/bin/xbmgmt examine --device 0000:41:00.0
sudo /opt/xilinx/xrt/bin/xbmgmt program --base --device 0000:41:00.0 --image xilinx_u55c_gen3x16_xdma_base_2
```


# how to start on acclnode02

```
./docker_run.sh xilinx/vitis-ai-cpu:2.5.0.1260
cd Vitis-AI
conda activate vitis-ai-pytorch
source /workspace/setup/alveo/setup.sh DPUCAHX8H-DWC
export XLNX_ENABLE_DEVICES=3
```


# ResNet50 cifar100 qat


For generate quantized model and deployment model

```
## The bitwidth can only be 16/8/4

## bitwidth=8 (notice the bitwidth need to be change in resnet50_cifar100_qat.py)
python resnet50_cifar100_qat.py
python resnet50_cifar100_qat.py --mode 'deploy'


# bitwidth=4
python resnet50_cifar100_qat.py --epochs 6 --quantizer_lr 1e-3 --weight_lr 1e-6
python resnet50_cifar100_qat.py --mode 'deploy'
```

For compiling DPU model

```
vai_c_xir -x qat_result/ResNet_0_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCAHX8H/U55C-DWC/arch.json -o dpu_xmodel_U55C -n resnet50_cifar100_U55C
```


For running the DPU model 

```
python resnet50_dpu.py 2  resnet50_qat_8bit/dpu_xmodel_U55C/resnet50_cifar100_U55C.xmodel
```


For profiling DPU model

```
cd /workspace/resnet50_cifar100/resnet50_qat_16bit
# the thread number has to be 2 or 3
python3 -m vaitrace_py ./resnet50_dpu.py 2 dpu_xmodel_U55C/resnet50_cifar100_U55C.xmodel
```

Then copy the *.csv file and xrt.run_summary to kw61088, run

```
vitis_analyzer xrt.run_summary
```


# ResNet18 cifar100 qat

Similarly,

```
# we can inspect the model
python resnet18_cifar100_quant.py --quant_mode float --inspect

# bitwidth=8
python resnet18_cifar100_qat.py
python resnet18_cifar100_qat.py --mode 'deploy'

# bitwidth=4
python resnet18_cifar100_qat.py --epochs 6 --quantizer_lr 1e-3 --weight_lr 1e-6
python resnet18_cifar100_qat.py --mode 'deploy'


# compile
vai_c_xir -x qat_result/ResNet_0_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCAHX8H/U55C-DWC/arch.json -o dpu_xmodel_U55C -n resnet18_cifar100_U55C


# run and profile
cd /workspace/resnet50_cifar100/resnet18_qat_16bit
python3 -m vaitrace_py ./resnet50_dpu.py 2 dpu_xmodel_U55C/res.xmodel

```


