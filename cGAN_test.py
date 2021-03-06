import os
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import numpy as np

# 하이퍼파라미터
epochs = 300
batch_size = 100
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
print('Using Device :',device)

# Fashion MNISt 데이터셋
trainset = datasets.FashionMNIST('./.data',
                                 train=True,
                                 download=True,
                                 transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,), (0.5,))
                                 ]))
train_loader = torch.utils.data.DataLoader(
    dataset = trainset,
    batch_size = batch_size,
    shuffle = True
)

# 생성자(Generator)
class Generator(nn.Module):
  def __init__(self):
    super().__init__()

    self.embed = nn.Embedding(10, 10)

    self.model = nn.Sequential(
        nn.Linear(110, 256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(256, 512),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(512, 1024),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(1024, 784),
        nn.Tanh()
    )

  def forward(self, z, labels):
    c = self.embed(labels)
    x = torch.cat([z, c], 1)
    return self.model(x)

# 판별자(Discriminator)
class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()

    self.embed = nn.Embedding(10, 10)

    self.model = nn.Sequential(
        nn.Linear(794, 1024),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(0.3), 
        nn.Linear(1024, 512),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(0.3), 
        nn.Linear(512, 256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(0.3), 
        nn.Linear(256, 128),   # default:(256,1)
        nn.LeakyReLU(0.2, inplace=True),
        nn.Dropout(0.3), 
        nn.Linear(128, 1),
        nn.Sigmoid()
    )

  def forward(self, x, labels):
    c = self.embed(labels)
    x = torch.cat([x, c], 1)
    return self.model(x)

# 모델 인스턴스를 만들고 모델의 가중치를 지정한 장치로 보내기
D = Discriminator().to(device)
G = Generator().to(device)

# 이진 교차 엔트로피 함수와
# 생성자와 판별자를 최적화할 Adam 모듈
criterion = nn.BCELoss()
d_optimizer = optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = optim.Adam(G.parameters(), lr=0.0002)

total_step = len(train_loader)
for epoch in range(1, epochs+1):
  for i, (images, labels) in enumerate(train_loader):
    images = images.reshape(batch_size, -1).to(device)

    # '진짜'와 '가짜' 레이블 생성
    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)

    # 판별자가 진짜 이미지를 진짜로 인식하는 오차 계산(데이터셋 레이블 입력)
    labels = labels.to(device)
    
    outputs = D(images, labels)
    d_loss_real = criterion(outputs, real_labels)
    real_score = outputs

    # 무작위 텐서와 무작위 레이블을 생성자에 입력해 가짜 이미지 생성
    z = torch.randn(batch_size, 100).to(device)
    g_label = torch.randint(0, 10, (batch_size,)).to(device)
    fake_images = G(z, g_label)

    # 판별자가 가짜 이미지를 가짜로 인식하는 오차 계산
    outputs = D(fake_images, g_label)
    d_loss_fake = criterion(outputs, fake_labels)
    fake_score = outputs
    
    # 진짜와 가짜 이미지를 갖고 낸 오차를 더해서 판별자의 오차 계산
    d_loss = d_loss_real + d_loss_fake

    # 역전파 알고리즘으로 판별자 모델의 학습을 지행
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()

    # 생성자가 판별자를 속였는지에 대한 오차 계산(무작위 레이블 입력)
    fake_images = G(z, g_label)
    outputs = D(fake_images, g_label)
    g_loss = criterion(outputs, real_labels)

    # 역전파 알고리즘으로 생성자 모델의 학습을 진행
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()
    g_loss.backward()
    g_optimizer.step()
  
  # save_image(fake_images.data[:25], "cgan_images/%d.png" % epoch, nrow=5, normalize=True)
  print('Epoch [{}/{}] d_loss: {:.4f} g_loss: {:.4f} D(x):{:.2f} D(G(z)): {:.2f}'.format(
                  epoch, epochs, d_loss.item(), g_loss.item(), real_score.mean().item(), fake_score.mean().item()
    ))

# 만들고 싶은 아이템 생성하고 시각화하기
z = torch.randn(1, 100).to(device)

for i in range(10):
  item_number = i
  g_label = torch.full((1,), item_number, dtype=torch.long).to(device)
  sample_images = G(z, g_label)
  sample_images_img = np.reshape(sample_images.data.cpu().numpy()[0], (28,28))
  plt.imshow(sample_images_img, cmap='gray')
  plt.show()