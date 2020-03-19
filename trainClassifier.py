import torch
import torchvision
import nonechucks
import visdom
import model
import lossManager
import progressbar
import data_augmentation
import radam

testtransform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(256),
    torchvision.transforms.ToTensor()
])
traindataset = torchvision.datasets.ImageFolder(root="/home/wilmot_p/DATA/CARS/folder/stanford-car-dataset-by-classes-folder/car_data/car_data/train/", transform=data_augmentation.traintransform)
testdataset = nonechucks.SafeDataset(torchvision.datasets.ImageFolder(root="/home/wilmot_p/DATA/CARS/folder/stanford-car-dataset-by-classes-folder/car_data/car_data/test/", transform=testtransform))
testdataloader = torch.utils.data.DataLoader(testdataset, batch_size=16, shuffle=True)
nb_classes = len(traindataset.classes)

viz = visdom.Visdom()

m = model.Model(nb_classes, 64)
m = m.cuda()
#m.load_state_dict(torch.load('classifier_model.pt'))
print(m)

initial_learning_rate = 100 / sum(p.numel() for p in m.parameters() if p.requires_grad)
print("Initail Learning rate", initial_learning_rate)
optim = radam.RAdam(m.parameters(), lr=initial_learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', factor=0.5, verbose=True)
#optim.load_state_dict(torch.load('classifier_optim.pt'))
trainlm = lossManager.LossManager(displayEvery = 100, win = "Train Losses")
testlm = lossManager.LossManager(displayEvery = 1, win = "Test Losses")

def train(m, optim, dataset):
    dataloader = torch.utils.data.DataLoader(nonechucks.SafeDataset(dataset), batch_size=16, shuffle=True, num_workers=16)
    m.train()
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    display = True
    for img, label in progressbar.progressbar(dataloader):
        if display:
            viz.images(img, win="TRAINING_SAMPLES", opts={'title':"Training Samples"})
            display = False
        img = img.cuda()
        label = label.cuda()
        optim.zero_grad()
        y = m(img).squeeze()
        _, max_indicices = torch.max(y, 1)
        confusion_matrix[max_indicices, label] += 1
        loss = torch.nn.functional.cross_entropy(y, label)
        trainlm.registerLoss(loss.item() / 16)
        loss.backward()
        optim.step()
    correct = torch.sum(confusion_matrix.diag()).item()
    total = torch.sum(confusion_matrix).item()
    viz.heatmap(confusion_matrix, win="ConfusionTrain", opts={
        'title': "CONFUSION TRAIN " + str(correct) + " / " + str(total) + " -- " + str(correct / total * 100) + "%",
        'rownames': traindataset.classes,
        'columnnames': traindataset.classes
    })

def test(m, dataloader):
    m.eval()
    confusion_matrix = torch.zeros(nb_classes, nb_classes)
    loss = 0
    display = True
    for img, label in progressbar.progressbar(dataloader):
        if display:
            viz.images(img, win="TEST_SAMPLES", opts={'title':"Testing Samples"})
            display = False
        img = img.cuda()
        label = label.cuda()
        y = m(img).squeeze()
        loss += torch.nn.functional.cross_entropy(y, label).item()
        _, max_indicices = torch.max(y, 1)
        confusion_matrix[max_indicices, label] += 1
    correct = torch.sum(confusion_matrix.diag()).item()
    total = torch.sum(confusion_matrix).item()
    viz.heatmap(confusion_matrix, win="ConfusionTest", opts={
        'title': "CONFUSION TEST " + str(correct) + " / " + str(total) + " -- " + str(correct / total * 100) + "%",
        'rownames': traindataset.classes,
        'columnnames': traindataset.classes
    })
    scheduler.step(loss)
    testlm.registerLoss(loss / len(testdataset))

while True:
    test(m, testdataloader)
    train(m, optim, traindataset)
    torch.save(m.state_dict(), "classifier_model.pt")
    torch.save(optim.state_dict(), "classifier_optim.pt")
