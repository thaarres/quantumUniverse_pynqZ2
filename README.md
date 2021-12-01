# Fast anomaly detection on a Xilinx Zynq SoC

In this tutorial, we will demonstrate how to design and deploy a fast and resource inexpensive anomaly detection algorithm on an FPGA.
In the first notebook we will design a quantized and pruned autoencoder, in the second notebook we will create the neccessary bitfile to deploy this model on an FPGA, and in the final notebook we will put the model on a small inexpensive chip and measure the latency.

## Set up environment
The Python environment neccessary to execute these notebooks are in the `environment.yml` file. To set it up using Conda:
```
conda env create -f environment.yml
conda activate ad_pynq
```