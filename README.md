# Alveo U55C Deployment: ResNet18/50 (CIFAR-100) and ResNet20 (CIFAR-10) with Vitis-AI & HLS(hls4ml) Comparison

The project deploys ResNet architectures on the **Xilinx Alveo U55C** using **`Vitis-AI`**, and compares the performance with a custom HLS accelerator using **`hls4ml`**.

* **⚡ Vitis-AI Deployment:**
    * Implementation of **ResNet18** and **ResNet50** on the **CIFAR-100** dataset.
    * Implementation of **ResNet20** on the **CIFAR-10** dataset.
    * Utilizes the Vitis-AI DPU (Deep Learning Processor Unit) overlay for high-throughput inference.

* **🛠️ HLS Custom Implementation:**
    * Implementation of a custom hardware accelerator of **ResNet20** using **`hls4ml`**.
    * Contrasts the **DPU accelerator** against a **custom FPGA accelerator**.
    * Source codes and synthesis scripts for this custom implementation are located in the `hls4ml/` folder.
