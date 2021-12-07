if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import transforms
    from PIL import Image

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 6, 5)
            self.pool = nn.MaxPool2d(2, 2)
            self.conv2 = nn.Conv2d(6, 16, 5)
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
            self.fc2 = nn.Linear(120, 84)
            self.fc3 = nn.Linear(84, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = torch.flatten(x, 1)     # flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x


    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    PATH = "cifar_net_model.pt"
    model = torch.load(PATH)
    model.eval()

    IMAGE_PATH = "Bird.png"
    img_pil = Image.open(IMAGE_PATH).convert('RGB')
    # print('Pillow: ', img_pil.mode, img_pil.size)

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((32, 32)),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img_tensor = transform(img_pil)
    img_tensor = img_tensor.unsqueeze(0)

    output = model(img_tensor)

    _, predicted = torch.max(output, 1)

    print('Predicted:', classes[predicted[0]])
