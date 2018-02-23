import torchvision.models as models
import torch
from torch.autograd import Variable
from utils.data_loader import get_data_loader
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

'''
Running feature extraction part for GAN model extraction 
    cifar-10    $  python main.py --dataroot datasets/cifar --dataset cifar --load_D trained_models/dcgan/cifar/discriminator.pkl --load_G trained_models/dcgan/cifar/generator.pkl
'''

class FeatureExtractionTest():

    def __init__(self, train_loader, test_loader, cuda_flag, batch_size):
        self.train_loader = train_loader
        self.test_loader = test_loader
        print("Train length: {}".format(len(self.train_loader)))
        print("Test length: {}".format(len(self.test_loader)))
        self.batch_size = batch_size

        # Remove fully connected layer and extract 2048 vector as feautre representation of image
        self.model = models.resnet152(pretrained=True).cuda()
        self.model = torch.nn.Sequential(*list(self.model.children())[:-1])


    # Feature extraction test #1 flattening image
    def flatten_images(self):
        """
            Flattening image as image representation.
            Input is image and output is flattened self.channels*32*32 dimensional numpy array
        """
        x_train, y_train = [], []
        x_test, y_test = [], []

        # flatten pixels of train images
        for i, (images, labels) in enumerate(self.train_loader):
            if i == len(self.train_loader) // self.batch_size:
                break
            images = images.numpy()
            labels = labels.numpy()

            # Iterate over batch and save as numpy array features of images and label
            for j in range(self.batch_size):
                x_train.append(images[j].flatten())
                y_train.append(labels[j])

        for i, (images, labels) in enumerate(self.test_loader):
            if i == len(self.test_loader) // self.batch_size:
                break

            images = images.numpy()
            labels = labels.numpy()

            # Iterate over batch and save as numpy array features of images and label
            for j in range(self.batch_size):
                x_test.append(images[j].flatten())
                y_test.append(labels[j])

        return x_train, y_train, x_test, y_test

    # Feature extraction test #4 transfer learning Inception v3 model pretrained
    # Resize imaged to 224x224 for pretrained models
    def inception_feature_extraction(self):
        """
            Extract features from images with pretrained ResNet152 on ImageNet, with removed fully-connected layer.
            Input is image and output is flattened 2048 dimensional numpy array
        """
        x_train, y_train = [], []
        x_test, y_test = [], []

        for i, (images, labels) in enumerate(self.train_loader):
            if i == len(self.train_loader) // self.batch_size:
                break

            images = Variable(images).cuda()

            # Feature extraction with Resnet152 resulting with feature vector of 2048 dimension
            outputs = self.model(images)

            # Convert FloatTensors to numpy array
            features = outputs.data.cpu().numpy()
            labels = labels.numpy()

            # Iterate over batch and save as numpy array features of images and label
            for j in range(self.batch_size):
                x_train.append(features[j].flatten())
                y_train.append(labels[j])


        for i, (images, labels) in enumerate(self.test_loader):
            if i == len(self.test_loader) // self.batch_size:
                break

            images = Variable(images).cuda()

            # Feature extraction with Resnet152 resulting with feature vector of 2048 dimension
            outputs = self.model(images)

            # Convert FloatTensors to numpy array
            features = outputs.data.cpu().numpy()
            labels = labels.numpy()

            # Iterate over batch and save as numpy array features of images and label
            for j in range(self.batch_size):
                x_test.append(features[j].flatten())
                y_test.append(labels[j])

        return x_train, y_train, x_test, y_test

    # Feature extraction GAN model discriminator output 1024x4x4
    def GAN_feature_extraction(self, discriminator):
        """
            Extract features from images with trained discriminator of GAN model.
            Input is image and output is flattened 16348 dimensional numpy array (1024x4x4)
            discriminator -- Trained discriminator of GAN model
        """
        x_train, y_train = [], []
        x_test, y_test = [], []
        for i, (images, labels) in enumerate(self.train_loader):
            if i == len(self.train_loader) // self.batch_size:
                break

            images = Variable(images).cuda()
            # Feature extraction DCGAN discriminator output 1024x4x4
            outputs = discriminator.feature_extraction(images)

            # Convert FloatTensors to numpy array
            features = outputs.data.cpu().numpy()
            labels = labels.numpy()

            # Iterate over batch and save as numpy array features of images and label
            for j in range(self.batch_size):
                x_train.append(features[j].flatten())
                y_train.append(labels[j])

        for i, (images, labels) in enumerate(self.test_loader):
            if i == len(self.test_loader) // self.batch_size:
                break

            images = Variable(images).cuda()
            outputs = discriminator.feature_extraction(images)

            # Convert FloatTensors to numpy array
            features = outputs.data.cpu().numpy()
            labels = labels.numpy()

            # Iterate over batch and save as numpy array features of images and label
            for j in range(self.batch_size):
                x_test.append(features[j].flatten())
                y_test.append(labels[j])

        return x_train, y_train, x_test, y_test


    def calculate_score(self):
        """
            Calculate accuracy score by fitting feature representation on to a linear classificato LinearSVM or LogisticRegression
        """
        mean_score = 0
        for i in range(10):
            # This way data is shuffling every iteration
            train_loader, test_loader = get_data_loader(args)

            x_train, y_train, x_test, y_test = feature_extraction.inception_feature_extraction()
            # x_train, y_train, x_test, y_test = feature_extraction.GAN_feature_extraction(model.D)
            # x_train, y_train, x_test, y_test = feature_extraction.flatten_images()

            # clf = LinearSVC()
            clf = LogisticRegression()
            clf.fit(x_train, y_train)

            predicted = clf.predict(x_test)
            score = accuracy_score(y_test, predicted)
            print("Accuaracy score: {}".format(score))
            mean_score += score
        print("Mean score: {}".format(float(mean_score) / float(10)))
        return float(mean_score) / float(10)
