import torch

class ContextBlock(torch.nn.Module):
    def __init__(self, inputChannels, outputChannels, stride):
        super(ContextBlock, self).__init__()
        assert(outputChannels % 8 == 0)
        self.a = torch.nn.Conv2d(inputChannels, outputChannels // 8 * 4, 3, stride=stride, padding=1, dilation=1)
        self.b = torch.nn.Conv2d(inputChannels, outputChannels // 8 * 2, 3, stride=stride, padding=2, dilation=2)
        self.c = torch.nn.Conv2d(inputChannels, outputChannels // 8 * 1, 3, stride=stride, padding=4, dilation=4)
        self.d = torch.nn.Conv2d(inputChannels, outputChannels // 8 * 1, 3, stride=stride, padding=8, dilation=8)

    def forward(self, x):
        return torch.cat([self.a(x), self.b(x), self.c(x), self.d(x)], 1)

if __name__ == "__main__":
    print("Stride 1")
    m = ContextBlock(3, 8, 1)
    print(m(torch.randn(1, 3, 512, 512)).shape)

    print("Stride 2")
    m = ContextBlock(3, 8, 2)
    print(m(torch.randn(1, 3, 512, 512)).shape)

