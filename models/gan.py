import os
import time
import torch
import torch.nn as nn
from torchvision import utils
from torch.autograd import Variable
from utils.tensorboard_logger import Logger


class GAN(object):
    def __init__(self, args):
        # Generator architecture
        self.G = nn.Sequential(
            nn.Linear(100, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Tanh())

        # Discriminator architecture
        self.D = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid())

        self.cuda = False
        self.cuda_index = 0
        # check if cuda is available
        self.check_cuda(args.cuda)

        # Binary cross entropy loss and optimizer
        self.loss = nn.BCELoss()
        self.d_optimizer = torch.optim.Adam(self.D.parameters(), lr=0.0002, weight_decay=0.00001)
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), lr=0.0002, weight_decay=0.00001)

        # Set the logger
        self.logger = Logger('./logs')
        self.number_of_images = 10
        self.epochs = args.epochs
        self.batch_size = args.batch_size

    # Cuda support
    def check_cuda(self, cuda_flag=False):
        if cuda_flag:
            self.cuda_index = 0
            self.cuda = True
            self.D.cuda(self.cuda_index)
            self.G.cuda(self.cuda_index)
            self.loss = nn.BCELoss().cuda(self.cuda_index)
            print("Cuda enabled flag: ")
            print(self.cuda)

    def train(self, train_loader):
        self.t_begin = time.time()
        generator_iter = 0

        for epoch in range(self.epochs+1):
            for i, (images, _) in enumerate(train_loader):
                # Check if round number of batches
                if i == train_loader.dataset.__len__() // self.batch_size:
                    break

                # Flatten image 1,32x32 to 1024
                images = images.view(self.batch_size, -1)
                z = torch.rand((self.batch_size, 100))

                if self.cuda:
                    real_labels = Variable(torch.ones(self.batch_size)).cuda(self.cuda_index)
                    fake_labels = Variable(torch.zeros(self.batch_size)).cuda(self.cuda_index)
                    images, z = Variable(images.cuda(self.cuda_index)), Variable(z.cuda(self.cuda_index))
                else:
                    real_labels = Variable(torch.ones(self.batch_size))
                    fake_labels = Variable(torch.zeros(self.batch_size))
                    images, z = Variable(images), Variable(z)

                # Train discriminator
                # compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
                # [Training discriminator = Maximizing discriminator being correct]
                outputs = self.D(images)
                d_loss_real = self.loss(outputs.flatten(), real_labels)
                real_score = outputs

                # Compute BCELoss using fake images
                fake_images = self.G(z)
                outputs = self.D(fake_images)
                d_loss_fake = self.loss(outputs.flatten(), fake_labels)
                fake_score = outputs

                # Optimizie discriminator
                d_loss = d_loss_real + d_loss_fake
                self.D.zero_grad()
                d_loss.backward()
                self.d_optimizer.step()

                # Train generator
                if self.cuda:
                    z = Variable(torch.randn(self.batch_size, 100).cuda(self.cuda_index))
                else:
                    z = Variable(torch.randn(self.batch_size, 100))
                fake_images = self.G(z)
                outputs = self.D(fake_images)

                # We train G to maximize log(D(G(z))[maximize likelihood of discriminator being wrong] instead of
                # minimizing log(1-D(G(z)))[minizing likelihood of discriminator being correct]
                # From paper  [https://arxiv.org/pdf/1406.2661.pdf]
                g_loss = self.loss(outputs.flatten(), real_labels)

                # Optimize generator
                self.D.zero_grad()
                self.G.zero_grad()
                g_loss.backward()
                self.g_optimizer.step()
                generator_iter += 1


                if ((i + 1) % 100) == 0:
                    print("Epoch: [%2d] [%4d/%4d] D_loss: %.8f, G_loss: %.8f" %
                          ((epoch + 1), (i + 1), train_loader.dataset.__len__() // self.batch_size, d_loss.data, g_loss.data))

                    if self.cuda:
                        z = Variable(torch.randn(self.batch_size, 100).cuda(self.cuda_index))
                    else:
                        z = Variable(torch.randn(self.batch_size, 100))

                    # ============ TensorBoard logging ============#
                    # (1) Log the scalar values
                    info = {
                        'd_loss': d_loss.data,
                        'g_loss': g_loss.data
                    }

                    for tag, value in info.items():
                        self.logger.scalar_summary(tag, value, i + 1)

                    # (2) Log values and gradients of the parameters (histogram)
                    for tag, value in self.D.named_parameters():
                        tag = tag.replace('.', '/')
                        self.logger.histo_summary(tag, self.to_np(value), i + 1)
                        self.logger.histo_summary(tag + '/grad', self.to_np(value.grad), i + 1)

                    # (3) Log the images
                    info = {
                        'real_images': self.to_np(images.view(-1, 32, 32)[:self.number_of_images]),
                        'generated_images': self.generate_img(z, self.number_of_images)
                    }

                    for tag, images in info.items():
                        self.logger.image_summary(tag, images, i + 1)


                if generator_iter % 1000 == 0:
                    print('Generator iter-{}'.format(generator_iter))
                    self.save_model()

                    if not os.path.exists('training_result_images/'):
                        os.makedirs('training_result_images/')

                    # Denormalize images and save them in grid 8x8
                    if self.cuda:
                        z = Variable(torch.randn(self.batch_size, 100).cuda(self.cuda_index))
                    else:
                        z = Variable(torch.randn(self.batch_size, 100))
                    samples = self.G(z)
                    samples = samples.mul(0.5).add(0.5)
                    samples = samples.data.cpu()
                    grid = utils.make_grid(samples)
                    utils.save_image(grid, 'training_result_images/gan_image_iter_{}.png'.format(
                        str(generator_iter).zfill(3)))

        self.t_end = time.time()
        print('Time of training-{}'.format((self.t_end - self.t_begin)))
        # Save the trained parameters
        self.save_model()

    def evaluate(self, test_loader, D_model_path, G_model_path):
        self.load_model(D_model_path, G_model_path)
        if self.cuda:
            z = Variable(torch.randn(self.batch_size, 100).cuda(self.cuda_index))
        else:
            z = Variable(torch.randn(self.batch_size, 100))
        samples = self.G(z)
        samples = samples.mul(0.5).add(0.5)
        samples = samples.data.cpu()
        grid = utils.make_grid(samples)
        print("Grid of 8x8 images saved to 'gan_model_image.png'.")
        utils.save_image(grid, 'gan_model_image.png')

    def generate_img(self, z, number_of_images):
        samples = self.G(z).data.cpu().numpy()[:number_of_images]
        generated_images = []
        for sample in samples:
            generated_images.append(sample.reshape(32,32))
        return generated_images

    def to_np(self, x):
        return x.data.cpu().numpy()

    def save_model(self):
        torch.save(self.G.state_dict(), './generator.pkl')
        torch.save(self.D.state_dict(), './discriminator.pkl')
        print('Models save to ./generator.pkl & ./discriminator.pkl ')

    def load_model(self, D_model_filename, G_model_filename):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.D.load_state_dict(torch.load(D_model_path))
        self.G.load_state_dict(torch.load(G_model_path))
        print('Generator model loaded from {}.'.format(G_model_path))
        print('Discriminator model loaded from {}-'.format(D_model_path))
