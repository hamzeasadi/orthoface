import torch
from torch import nn as nn
from torch import functional as F
from torchvision import models
# from torchinfo import summary

class OrthoMetric(nn.Module):
    """
    doc
    """
    def __init__(self):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT
        self.resnet = models.resnet50(weights=weights)
        for param in self.resnet.parameters():
            param.requires_grad = False

        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        self.resnet.fc = nn.Linear(in_features=2048, out_features=256)
        self.th = nn.Tanh()

    def forward(self, x):
        x = self.resnet(x)

        return self.th(x)




def main():
    model = OrthoMetric()
    x = torch.randn(size=(10, 3, 224, 224))
    # summary(model=model, input_size=[1, 3, 224, 224], col_names=['kernel_size', 'output_size', 'num_params', "mult_adds"], row_settings=['var_names'])
    # out = model(x)
    # print(out.shape)


if __name__ =='__main__':
    main()