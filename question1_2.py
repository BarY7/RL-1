import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

def question1_2():
    # Hyper Parameters
    input_size = 784
    num_classes = 10
    num_epochs = 100
    batch_size = 600
    learning_rate = 1e-1

    # MNIST Dataset
    train_dataset = dsets.MNIST(root='./data',
                                train=True,
                                transform=transforms.ToTensor(),
                                download=True)

    test_dataset = dsets.MNIST(root='./data',
                            train=False,
                            transform=transforms.ToTensor())

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=batch_size,
                                            shuffle=False)


    # Neural Network Model
    class Net(nn.Module):
        def __init__(self, input_size, num_classes):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(input_size, num_classes)

        def forward(self, x):
            out = self.fc1(x)
            return out

    net = Net(input_size, num_classes)

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    loss_epoch_array = []

    # Train the Model
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            # Convert torch tensor to Variable
            images = Variable(images.view(-1, 28*28))
            labels = Variable(labels)

            # Forward + Backward + Optimize
            # TODO: implement training code
            pred = net(images)
            loss = criterion(pred,labels)
            print("loss at {} batch is: {}".format(i, loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        loss_epoch_array.append(loss.item())

    # Test the Model
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images.view(-1, 28*28))
        # TODO: implement evaluation code - report accuracy
        pred = net(images)
        correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
        total += labels.size(0)

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))

    # Save the Model
    torch.save(net.state_dict(), 'model.pkl')
    return (loss_epoch_array,(100 * correct / total))