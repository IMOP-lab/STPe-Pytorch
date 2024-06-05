# STPe-Pytorch:Stratified Trio-dimensional Pathway Encoding Framework

STPe: Stratified Trio-dimensional Pathway Encoding Framework

Minhong Sun, Han Yang, Jian Huang, Yihao Guo, Gaopeng Huang, Xuanbin Chen, Duanpo Wu, Shaowei Jiang, Xiaoshuai Zhang, Hong He, Xingru Huang, and Guan Gui

Hangzhou Dianzi University IMOP-lab

<div align=center>
  <img src="https://github.com/IMOP-lab/STPe-Pytorch/blob/main/images/HeON.png">
</div>
<p align=center>
  Figure 1: The overarching framework of the proposed Hybrid-distributed Elastic Ophthalmic Navigation System (HeON) for real-time macular surgery guidance.
</p>

We propose the HeON system, based on the IoMT, an efficient intraoperative stable delivery of quantization-aid 3D reconstruction system.

Next, we introduce our innovative STPe Segmention framework


<div align=center>
  <img src="https://github.com/IMOP-lab/STPe-Pytorch/blob/main/images/STPe.png">
</div>
<p align=center>
  Figure 2: The overall structure of proposed Stratified Trio-dimensional Pathway Encoding framework.
</p>


SpectroConv Discriminative Synthesis Encoder depicted in Figure 3, initially processes the raw input image in parallel through convolution, Fast Fourier Transform (FFT), and Discrete Wavelet Transform (DWT) operations. Convolutional networks describe spatial details. FFT transitions from the time domain to the frequency domain. By analyzing the global periodic structure and frequency characteristics, it is used to identify and analyze background noise and fine-grain texture differences. Wavelet transform provides multi-scale time-frequency analysis, capturing high frequency, short term and low frequency and long term information for detecting local changes in image features. The encoder also combines attention mechanisms in the frequency and time-frequency domains to refine and emphasize key features within these regions.

<div align=center>
  <img src="https://github.com/IMOP-lab/STPe-Pytorch/blob/main/images/SpectroConv.png">
</div>
<p align=center>
  Figure 3: SpectroConv for extracting intricate time-frequency domain features and nuanced frequency domain information from image data.
</p>
