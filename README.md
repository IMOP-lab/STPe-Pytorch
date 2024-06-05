# STPe-Pytorch:Stratified Trio-dimensional Pathway Encoding Framework

STPe: Stratified Trio-dimensional Pathway Encoding Framework

Minhong Sun, Han Yang, Jian Huang, Yihao Guo, Gaopeng Huang, Xuanbin Chen, Duanpo Wu, Shaowei Jiang, Xiaoshuai Zhang, Hong He, Xingru Huang, and Guan Gui

Hangzhou Dianzi University IMOP-lab

## Methods
### STPe
<div align=center>
  <img src="https://github.com/IMOP-lab/STPe-Pytorch/blob/main/images/HeON.png">
</div>
<p align=center>
  Figure 1: The HeON.overarching framework of the proposed Hybrid-distributed Elastic Ophthalmic Navigation System (HeON) for real-time macular surgery guidance.
</p>

We propose the HeON system, based on the IoMT, an efficient intraoperative stable delivery of quantization-aid 3D reconstruction system.

Next, we introduce our innovative STPe Segmention framework

<div align=center>
  <img src="https://github.com/IMOP-lab/STPe-Pytorch/blob/main/images/STPe.png">
</div>
<p align=center>
  Figure 2: The overall structure of proposed Stratified Trio-dimensional Pathway Encoding framework.
</p>

#### SpectroConv Module

<div align=center>
  <img src="https://github.com/IMOP-lab/STPe-Pytorch/blob/main/images/SpectroConv.png">
</div>
<p align=center>
  Figure 3: SpectroConv Module.
</p>

The SpectroConv Module for extracting intricate time-frequency domain features and nuanced frequency domain information from image data.

#### HIT-EAFS Module

<div align=center>
  <img src="https://github.com/IMOP-lab/STPe-Pytorch/blob/main/images/HIT-EAFS.png">
</div>
<p align=center>
  Figure 4: HIT-EAFS Module.
</p>

HIT-EAFS Module is designed to address the nuanced requirements of capturing both global and local feature representations effectively.

## Experiment
### Baselines
### Training Results

<div align=center>
    <img src="https://github.com/IMOP-lab/STPe-Pytorch/blob/main/table/baseline.png">
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

