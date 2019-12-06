# Learned-Quantized-SR(FSRCNN)-Tensorpack
Super Resolution FSRCNN with [Learned Quantized](https://arxiv.org/abs/1807.10029) activation, weights. <br />

heavily referenced on :<br />
https://github.com/microsoft/LQ-Nets <br />
https://github.com/Saafke/FSRCNN_Tensorflow <br />
https://github.com/tensorpack/tensorpack/tree/master/examples/SuperResolution <br />
[tensorpack api doc](https://tensorpack.readthedocs.io/) <br />

 
### Dependencies
+ Python 2.7 or 3.3+
+ Python bindings for OpenCV
+ TensorFlow >= 1.3.0
+ [TensorPack](https://github.com/tensorpack/tensorpack)

### Learned Quantized
Based on [LQ-Net paper](https://arxiv.org/abs/1807.10029) and its [code](https://github.com/microsoft/LQ-Nets).

### Cholesky Decomposition to replace matrix inverse
Slight modification to original LQ-Net [Conv2dQuant layer](https://github.com/microsoft/LQ-Nets/blob/master/learned_quantization.py)


