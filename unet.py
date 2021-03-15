import torch
import torch.nn as nn
import torch.optim as optim
import os
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

def double_conv(in_c,out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size = 3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c, out_c, kernel_size = 3),
        nn.ReLU(inplace=True)
    )
    return conv

def crop_img(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta = (tensor_size - target_size) // 2
    if(tensor_size%2 != 0):
        return tensor[:,:,delta+1:tensor_size - delta, delta+1:tensor_size -delta]
    else:
        return tensor[:,:,delta:tensor_size - delta, delta:tensor_size -delta]


class mydataset(torch.utils.data.Dataset):


    def __init__(self, input_path, target_path, length):
        super(mydataset, self).__init__()
        self.input_path = input_path
        self.target_path = target_path
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        idx = idx * 500
        patch_idx = idx % 64
        if patch_idx // 8 == 0 or patch_idx // 8 == 7:
            idx += 8
        patch_idx = idx % 64
        if patch_idx % 8 == 0 or patch_idx % 8 == 7:
            idx += 1
        input_loadpath = self.input_path + str(idx + 1) + '.mat'
        input_image = torch.tensor(sio.loadmat(input_loadpath)['x'])
        target_loadpath = self.target_path + str(idx + 1) + '.mat'
        target_image = torch.tensor(sio.loadmat(target_loadpath)['x'])
        input_image = input_image.float()
        target_image = target_image.float()
        if len(input_image.size()) == 2:
            input_image.unsqueeze_(0)
        if len(target_image.size()) == 2:
            target_image.unsqueeze_(0)

        return input_image, target_image


class UNet(nn.Module):


    def __init__(self):
        super(UNet, self).__init__()

        self.max_pool_2x2 = nn.MaxPool2d(kernel_size = 2, stride = 2)
        self.down_conv_1 = double_conv(1,64)
        self.down_conv_2 = double_conv(64,128)
        self.down_conv_3 = double_conv(128,256)
        self.down_conv_4 = double_conv(256,512)
        self.down_conv_5 = double_conv(512,1024)

        self.up_trans_1 = nn.ConvTranspose2d(
            in_channels=1024,
            out_channels=512,
            kernel_size=2,
            stride=2)
        self.up_conv_1 = double_conv(1024,512)

        self.up_trans_2 = nn.ConvTranspose2d(
            in_channels=512,
            out_channels=256,
            kernel_size=2,
            stride=2)
        self.up_conv_2 = double_conv(512,256)

        self.up_trans_3 = nn.ConvTranspose2d(
            in_channels=256,
            out_channels=128,
            kernel_size=2,
            stride=2)
        self.up_conv_3 = double_conv(256,128)

        self.up_trans_4 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2)
        self.up_conv_4 = double_conv(128,64)

        self.out = nn.Conv2d(
        in_channels = 64,
        out_channels = 2,
        kernel_size = 1
        )



    def forward(self, image):

        #encoder
         x1 = self.down_conv_1(image)
         x2 = self.max_pool_2x2(x1)
         x3 = self.down_conv_2(x2)
         x4 = self.max_pool_2x2(x3)
         x5 = self.down_conv_3(x4)
         x6 = self.max_pool_2x2(x5)
         x7 = self.down_conv_4(x6)
         x8 = self.max_pool_2x2(x7)
         x9 = self.down_conv_5(x8)

         #decoder
         x = self.up_trans_1(x9)
         y = crop_img(x7,x)
         x = self.up_conv_1(torch.cat([x,y],1))

         x = self.up_trans_2(x)
         y = crop_img(x5,x)
         x = self.up_conv_2(torch.cat([x,y],1))

         x = self.up_trans_3(x)
         y = crop_img(x3,x)
         x = self.up_conv_3(torch.cat([x,y],1))

         x = self.up_trans_4(x)
         y = crop_img(x1,x)
         x = self.up_conv_4(torch.cat([x,y],1))

         x = self.out(x)
         return x

def train(dataset):
    epochs = 10
    model = UNet()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):
        running_loss = 0.0
        for i in range(len(dataset)):
            input,target = dataset[i]
            input = input.reshape((1,1,320,320))
            target = target.reshape((1,1,320,320))
            optimizer.zero_grad()

            # forward + backward + optimize
            output = model(input)

            print(output.size())
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')


if __name__ == "__main__":
    #image = torch.rand(1,1,320,320)
    input_path = "output/output300/"
    target_path = "output/output600/"
    data = mydataset(input_path,target_path,500)
    train(data)
