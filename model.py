import torch
import context_block

class Model(torch.nn.Module):
    def __init__(self, nb_classes, nc=16):
        super(Model, self).__init__()

        self.features = torch.nn.Sequential(
            # 3x256x256
            torch.nn.Dropout(.2),
            context_block.ContextBlock(3, nc, 1),
            torch.nn.CELU(),
            torch.nn.BatchNorm2d(nc),
            # ncx256x256
            torch.nn.Dropout(.2),
            context_block.ContextBlock(nc, 2*nc, 2),
            torch.nn.CELU(),
            torch.nn.BatchNorm2d(2*nc),
            # 2*ncx128x128
            torch.nn.Dropout(.2),
            context_block.ContextBlock(2*nc, nc, 1),
            torch.nn.CELU(),
            torch.nn.BatchNorm2d(nc),
            # ncx128x128
            torch.nn.Dropout(.2),
            context_block.ContextBlock(nc, 2*nc, 2),
            torch.nn.CELU(),
            torch.nn.BatchNorm2d(2*nc),
            # 2*ncx2*ncx2*nc
            torch.nn.Dropout(.2),
            context_block.ContextBlock(2*nc, nc, 1),
            torch.nn.CELU(),
            torch.nn.BatchNorm2d(nc),
            # ncx2*ncx2*nc
            torch.nn.Dropout(.2),
            context_block.ContextBlock(nc, 2*nc, 2),
            torch.nn.CELU(),
            torch.nn.BatchNorm2d(2*nc),
            # 2*ncxncxnc
            torch.nn.Dropout(.2),
            context_block.ContextBlock(2*nc, nc, 1),
            torch.nn.CELU(),
            torch.nn.BatchNorm2d(nc),
            # ncxncxnc
            torch.nn.Dropout(.2),
            context_block.ContextBlock(nc, 2*nc, 2),
            torch.nn.CELU(),
            torch.nn.BatchNorm2d(2*nc),
            # 2*ncx16x16
            torch.nn.Dropout(.2),
            context_block.ContextBlock(2*nc, nc, 1),
            torch.nn.CELU(),
            torch.nn.BatchNorm2d(nc),
            # ncx16x16
            torch.nn.Dropout(.2),
            context_block.ContextBlock(nc, 2*nc, 2),
            torch.nn.CELU(),
            torch.nn.BatchNorm2d(2*nc),
            # 2*ncx8x8
            torch.nn.Dropout(.2),
            context_block.ContextBlock(2*nc, nc, 1),
            torch.nn.CELU(),
            torch.nn.BatchNorm2d(nc),
            # ncx8x8
            )
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Dropout(.2),
            torch.nn.Linear(nc*8*8, 100),
            torch.nn.CELU(),
            torch.nn.Dropout(.2),
            torch.nn.Linear(100, nb_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
