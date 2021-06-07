## Pytorch code for GAN models
This is the pytorch implementation of 3 different GAN models using same convolutional architecture.


- DCGAN (Deep convolutional GAN)
- WGAN-CP (Wasserstein GAN using weight clipping)
- WGAN-GP (Wasserstein GAN using gradient penalty)



## Dependecies
The prominent packages are:

* numpy
* scikit-learn
* tensorflow 2.5.0
* pytorch 1.8.1
* torchvision 0.9.1

To install all the dependencies quickly and easily you should use __pip__

```python
pip install -r requirements.txt
```



 *Training*
 ---
Running training of DCGAN model on Fashion-MNIST dataset:


```
python main.py --model DCGAN \
               --is_train True \
               --download True \
               --dataroot datasets/fashion-mnist \
               --dataset fashion-mnist \
               --epochs 30 \
               --cuda True \
               --batch_size 64
```

Running training of WGAN-GP model on CIFAR-10 dataset:

```
python main.py --model WGAN-GP \
               --is_train True \
               --download True \
               --dataroot datasets/cifar \
               --dataset cifar \
               --generator_iters 40000 \
               --cuda True \
               --batch_size 64
```

Start tensorboard:

```
tensorboard --logdir ./logs/
```

*Walk in latent space*
---
*Interpolation between a two random latent vector z over 10 random points, shows that generated samples have smooth transitions.*


<img src="images/latent_fashion.png" width="350"> &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;   <img src="images/latent-mnist.png" width="350">





*Generated examples MNIST, Fashion-MNIST, CIFAR-10*
---

<img src="images/CIFAR-10.png" width="800">


<img src="images/Fashion-MNIST.png" width="770">

<img src="images/MNIST.png" width="800">



*Inception score*
---
  [About Inception score](https://arxiv.org/pdf/1801.01973.pdf)


<img src="images/inception_graph_generator_iters.png" width="400" > &nbsp; &nbsp;&nbsp; &nbsp;&nbsp; &nbsp;   <img src="images/inception_graph_time.png" width="400">


*Useful Resources*
---


- [WGAN reddit thread](https://www.reddit.com/r/MachineLearning/comments/5qxoaz/r_170107875_wasserstein_gan/)
- [Blogpost](https://lilianweng.github.io/lil-log/2017/08/20/from-GAN-to-WGAN.html)
- [Deconvolution and checkboard Artifacts](https://distill.pub/2016/deconv-checkerboard/)
- [WGAN-CP paper](https://arxiv.org/pdf/1701.07875.pdf)
- [WGAN-GP paper](https://arxiv.org/pdf/1704.00028.pdf)
- [DCGAN paper](https://arxiv.org/pdf/1511.06434.pdf)
- [Working remotely with PyCharm and SSH](https://medium.com/@erikhallstrm/work-remotely-with-pycharm-tensorflow-and-ssh-c60564be862d)
