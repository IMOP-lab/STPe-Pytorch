# STPe-Pytorch:Stratified Trio-dimensional Pathway Encoding Framework

STPe: Stratified Trio-dimensional Pathway Encoding Framework

Minhong Sun, Han Yang, Jian Huang, Yihao Guo, Gaopeng Huang, Xuanbin Chen, Duanpo Wu, Shaowei Jiang, Xiaoshuai Zhang, Hong He, Xingru Huang, and Guan Gui

Hangzhou Dianzi University IMOP-lab

## Methods
### STPe
<div align=center>
  <img src="https://github.com/IMOP-lab/STPe-Pytorch/blob/main/images/HeON.png"width=70% height=70%>
</div>
<p align=center>
  Figure 1: The HeON.overarching framework of the proposed Hybrid-distributed Elastic Ophthalmic Navigation System (HeON) for real-time macular surgery guidance.
</p>

We propose the HeON system, based on the IoMT, an efficient intraoperative stable delivery of quantization-aid 3D reconstruction system.

Next, we introduce our innovative STPe Segmention framework

<div align=center>
  <img src="https://github.com/IMOP-lab/STPe-Pytorch/blob/main/images/STPe.png"width=70% height=70%>
</div>
<p align=center>
  Figure 2: The overall structure of proposed Stratified Trio-dimensional Pathway Encoding framework.
</p>

#### SpectroConv Module

<div align=center>
  <img src="https://github.com/IMOP-lab/STPe-Pytorch/blob/main/images/SpectroConv.png"width=70% height=70%>
</div>
<p align=center>
  Figure 3: SpectroConv Module.
</p>

The SpectroConv Module for extracting intricate time-frequency domain features and nuanced frequency domain information from image data.

#### HIT-EAFS Module

<div align=center>
  <img src="https://github.com/IMOP-lab/STPe-Pytorch/blob/main/images/HIT-EAFS.png"width=70% height=70%>
</div>
<p align=center>
  Figure 4: HIT-EAFS Module.
</p>

HIT-EAFS Module is designed to address the nuanced requirements of capturing both global and local feature representations effectively.

## Experiment
### Baselines
### Training Results

<div align=center>
    <img src="https://github.com/IMOP-lab/STPe-Pytorch/blob/main/table/baselinedata.png">
</div>
<p align=center>
  Figure 5: Compare the results with other models
</p>

The proposed STPe segmentation results are compared with previous segmentation models, with evaluations across five categories: All (average of all categories), MH (Macular Hole), ME (Macular Edema), RA (Retina), and CR (Choroid). The best values for each metric are displayed in red, the second-best values in blue, and both the best and second-best metrics are indicated with highlights.


<div align=center>
    <img src="https://github.com/IMOP-lab/STPe-Pytorch/blob/main/images/baseline.png">
</div>
<p align=center>
  Figure 6: Visual segmentation performance 
</p>

In the OIMHS dataset, visually compare STPe with other 14 segmentation models. We have selected 8 representative sequences for display. In these images, the segmentation results of macular holes are shown in red, macular edema in blue, the retina in yellow, and the choroid in green.

### Ablation
### Training Results

<div align=center>
    <img src="https://github.com/IMOP-lab/STPe-Pytorch/blob/main/table/ablationdata.png">
</div>
<p align=center>
  Figure 7: Ablation
</p>

The ablation of key modules in STPe was studied in the OIMHS data set. LDE (ResNet), GLE (MaxVit), and SCE (SpectroConv Discriminative Synthesis Encoder) corresponded to three branches in the Tri-dimensional Pathway Encoding, and NMCS, HIT-EAFS, and DFM corresponded to three key modules. The assessment included five categories: All, MH, ME, RA, and CR. The best values for each metric are displayed in red, the second-best values in blue, and both the best and second-best metrics are indicated with highlights.

<div align=center>
    <img src="https://github.com/IMOP-lab/STPe-Pytorch/blob/main/images/Ablation.png">
</div>
<p align=center>
  Figure 8: Visual segmentation performance 
</p>

Ablation experiments were executed based on the OIMHS dataset to evaluate the proposed STPe methodology. A1-A5 are denoted for group A and B1-B4 for group B, respectively. Four representative sequences were selected. In the depicted images, macular holes are delineated in red, macular edema in blue, the retina in yellow, and the choroid in green.

### Reconstruction

<div align=center>
    <img src="https://github.com/IMOP-lab/STPe-Pytorch/blob/main/images/reconstruction.png"width=70% height=70%>
</div>
<p align=center>
  Figure 9: Reconstruction
</p>

Example of two corresponding reconstruction outcomes (right) with original OCT images (left). The outcome delineates architectures of opaque retina, the third panel elucidates the transparent condition, highlighting the macular hole (red region) and macular edema (dark blue region).


