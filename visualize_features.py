import torch
import torchvision
import model
import progressbar

m = model.Model(196)
m.load_state_dict(torch.load("classifier_model.pt"))
m = m.features.cuda()
m.eval()
print(m)

data = torch.Tensor(1, 3, 256, 256).cuda()
while len(m):
    while len(m) and str(type(m[-1])) != "<class 'torch.nn.modules.activation.CELU'>":
        m = m[:-1]
    print(m)
    channel = m(data).shape[1]
    for c in progressbar.progressbar(range(channel)):
        canvas = torch.randn(1, 3, 256, 256).cuda()
        canvas.requires_grad = True
        optim = torch.optim.Adam([canvas], lr=0.01)
        for _ in range(250):
            x = canvas
            optim.zero_grad()
            x = x.clamp(0, 1)
            y = m(x)
            l = -torch.mean(y[0][c])
            l.backward()
            optim.step()
        name = "FEATURES/{}_{}.png".format(len(m), c)
        torchvision.utils.save_image(canvas[0], name)
    m = m[:-2]
