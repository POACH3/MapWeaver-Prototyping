"""
grids_cnn.py

NOTES:
    - import image dataset
    - image processing (grayscale, threshold, differencing)
    - architecture
    - train
    - test

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
print(torch.__version__)
print(torch.backends.mps.is_available())

class GridCNN(nn.Module):

    def __init__(self):
        super().__init__()

        # structure


    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)

        x = F.relu(self.conv3(x))

        x = self.up1(x)
        x = self.up2(x)

        x = torch.sigmoid(self.out(x))

        return x






device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)

model = MyCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(10):
    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

    print("epoch done")