import torch
from torch import nn


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=3 // 2)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        return torch.cat([x, self.leaky_relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(*[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning

class FeatureReview(nn.Module):
    def __init__(self, in_channels):
        super(FeatureReview, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=3 // 2)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        return self.leaky_relu(self.conv(x))
        
class RDN(nn.Module):
    def __init__(self, num_channels, num_features, growth_rate, num_blocks, num_layers):
        super(RDN, self).__init__()
        #num_channels = 16
        self.G0 = num_features #64
        self.G = growth_rate #64
        self.D = num_blocks #16
        self.C = num_layers #8

        # shallow feature extraction
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=1)
        
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )
        self.conv = nn.Conv2d(self.G0,self.G0, kernel_size=3, padding=3 // 2)

        # output layer
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        #self.output = nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2)
        
        self.output = nn.Conv2d(self.G0, num_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        original = x
        sfe1 = self.sfe1(x)
        sfe1 = self.leaky_relu(sfe1)
        sfe2 = self.sfe2(sfe1)
        sfe2 = self.leaky_relu(sfe2)

        x = sfe2
        local_features = []
 
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) # global residual learning
        x = self.leaky_relu(x)
        x = self.conv(x)
        x = self.leaky_relu(x)
        out = sfe1 + x
        out = self.conv(out)
        out = self.leaky_relu(out)
        out = self.output(out)
        out = self.tanh(out)
        out = out + original



        return out # adding the residual back to the input
