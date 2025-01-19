import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gs
import time
import tqdm
import copy

from torch.utils.data import DataLoader
from tqdm import tqdm

# ----------------------------------------------------------------------------------------------------------------------
# We set the GPU if available
torch.cuda.empty_cache()

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("\n\n\n----- Running of a GPU -----\n")
else:
    device = torch.device("cpu")
    print("\n\n\n----- Running on a CPU -----\n")

# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- LOADING THE DATASET -----------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# We set the batch size
batch_size: int = 16

# We set the image dimension we are working on
img_dim: int = 64


# We make a custom transform (resize) of an image
class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.size
        if h > w:
            new_h, new_w = self.output_size * h / w, self.output_size
        else:
            new_h, new_w = self.output_size, self.output_size * w / h
        new_h, new_w = int(new_h), int(new_w)
        image = TF.resize(image, (new_w, new_h))

        return image


# We define the basic transformation over the images
input_transform = transforms.Compose([
    Rescale(img_dim),
    transforms.CenterCrop(img_dim),
    transforms.ToTensor()
])

target_transform = transforms.Compose([
    Rescale(img_dim),
    transforms.CenterCrop(img_dim),
    transforms.Lambda(lambda x: torch.from_numpy(np.array(x) - 1).long())
])

# We download the Oxford-III-Pet dataset for training and testing, respectively
# First, we set the path for these datasets
training_set_path = "C:\Datasets\Segmentation"
test_set_path = "C:\Datasets\Segmentation"

train_set = torchvision.datasets.OxfordIIITPet(
    root=training_set_path,
    download=True,
    target_types='segmentation',
    transform=input_transform,
    target_transform=target_transform
)

test_set = torchvision.datasets.OxfordIIITPet(
    root=test_set_path,
    download=True,
    target_types='segmentation',
    split='test',
    transform=input_transform,
    target_transform=target_transform
)

# We set the DataLoaders corresponding to each set
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)

# ----------------------------------------------------------------------------------------------------------------------
print("\n\n\n--- Just Checking ---\n")

num_samples = 5
img_grid = gs.GridSpec(num_samples, 4, width_ratios=[1, 1, 1, 1])
fig = plt.figure(figsize=(10, 10))
plt.axis('off')

for idx, i in enumerate(np.random.choice(len(train_set), num_samples)):
    image, label = train_set[i]
    plt.subplot(img_grid[idx, 0]).imshow(image.permute(1, 2, 0))
    plt.subplot(img_grid[idx, 1]).imshow(label.squeeze())

for ax in fig.get_axes():
    ax.tick_params(bottom=False, labelbottom=False, left=False, labelleft=False)

plt.show()


# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------ THE ARCHITECTURE ----------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class UNetBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(UNetBlock, self).__init__()

        self.convolutionalLayer1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=3,
            padding=(1, 1))
        self.reluActivation = nn.ReLU()
        self.batchNorm = nn.BatchNorm2d(output_channels)

        self.convolutionalLayer2 = nn.Conv2d(
            in_channels=output_channels,
            out_channels=output_channels,
            kernel_size=3,
            padding=(1, 1))

    def forward(self, x):
        out = self.convolutionalLayer1(x)
        out = self.reluActivation(out)
        out = self.batchNorm(out)

        out = self.convolutionalLayer2(out)
        out = self.reluActivation(out)
        out = self.batchNorm(out)

        return out


class UNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.num_classes = num_classes

        self.uNetBlock1 = UNetBlock(input_channels=3, output_channels=8)
        self.poolLayer1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.uNetBlock2 = UNetBlock(input_channels=8, output_channels=16)
        self.poolLayer2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.uNetBlock3 = UNetBlock(input_channels=16, output_channels=32)
        self.poolLayer3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.uNetBlock4 = UNetBlock(input_channels=32, output_channels=64)
        self.transposeConvLayer1 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1)

        self.uNetBlock5 = UNetBlock(input_channels=64, output_channels=32)
        self.transposeConvLayer2 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=16,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1)

        self.uNetBlock6 = UNetBlock(input_channels=32, output_channels=16)
        self.transposeConvLayer3 = nn.ConvTranspose2d(
            in_channels=16,
            out_channels=8,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1)

        self.convLayer = nn.Conv2d(
            in_channels=8,
            out_channels=num_classes,
            kernel_size=3,
            stride=1,
            padding=(1, 1))

    def forward(self, x):
        out1 = self.uNetBlock1(x)
        out2 = self.poolLayer1(out1)

        out2 = self.uNetBlock2(out2)
        out3 = self.poolLayer2(out2)

        out3 = self.uNetBlock3(out3)
        out4 = self.poolLayer3(out3)

        out4 = self.uNetBlock4(out4)

        out5 = self.transposeConvLayer1(out4)
        out5 = torch.cat([out3, out5], dim=1)
        out5 = self.uNetBlock5(out5)

        out6 = self.transposeConvLayer2(out5)
        out6 = torch.cat([out2, out6], dim=1)
        out6 = self.uNetBlock6(out6)

        out7 = self.transposeConvLayer3(out6)
        out = self.convLayer(out7)

        return out


# ----------------------------------------------------------------------------------------------------------------------
# -------------------------------------------- THE IOU METRIC ----------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
# We define the Intersection-Over-Union (IoU) metric, used for computing the performance of the u-net model

def IoU(outputs: torch.Tensor, labels: torch.Tensor):
    SMOOTH = 1e-5

    class_outputs = torch.argmax(outputs, dim=1)

    intersection = (class_outputs & labels).sum()
    union = (class_outputs | labels).sum()

    batch_iou = (intersection + SMOOTH) / (union + SMOOTH)

    return batch_iou


def Model_IoU(model, train_loader):
    model = model.to(device)  # Set the model to GPU if available

    iou_set = []
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)  # Set the data to GPU if available

        outputs = model(inputs)
        iou_set.append(IoU(outputs, labels))

    mean_iou = sum(iou_set) / len(iou_set)

    return mean_iou


# ----------------------------------------------------------------------------------------------------------------------
# ------------------------------------------ TRAINING ------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
def train(model, train_set, batch_size, num_epochs, optimizer):
    print("\n\n\n----- Begin the Training Process -----")
    start_time_train = time.perf_counter()

    model = model.to(device)  # Set the model to GPU if available
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True)

    best_iou = -np.inf
    best_weights = None
    best_epoch = 0

    train_loss = []
    for epoch in range(num_epochs):
        print(f"\n----- Begin training epoch {epoch + 1} -----")
        start_time_epoch = time.perf_counter()

        model.train()  # Set the model to training mode

        epoch_loss = 0
        for images, labels in tqdm(train_loader):
            images, labels = images.to(device), labels.to(device)  # Set the data to GPU if available

            # Forward and Backward passes
            predictions = model(images)
            optimizer.zero_grad()
            loss = F.cross_entropy(predictions, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        epoch_iou = Model_IoU(model, train_loader=train_loader)
        train_loss.append(epoch_loss / len(train_loader))

        if epoch_iou > best_iou:
            best_iou = epoch_iou
            best_weights = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

        end_time_epoch = time.perf_counter()
        time_epoch = end_time_epoch - start_time_epoch

        print(f"\nEpoch: {epoch + 1} --- Training Loss: {train_loss[-1]: .6} --- IOU: {epoch_iou: .6} --- "
              f"Time: {time_epoch: .2}  --- Best IoU: {best_iou: .6} obtained at epoch {best_epoch}")

    # Set the model('s weights) with the best IoU
    model.load_state_dict(best_weights)

    # Save the best model, based on the IoU metric
    path_best_model = "../unet-segmentation-basic/u_net_basic.pth"
    torch.save(model, path_best_model)

    end_time_train = time.perf_counter()
    time_train = end_time_train - start_time_train
    print(f"\nTime for model's training was: {time_train: .2}")

    return train_loss


# ----------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------  MAIN() --------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
num_epochs: int = 15
learning_rate = 1e-3

num_classes: int = 3
my_unet = UNet(num_classes=num_classes)

optimizer = optim.Adam(my_unet.parameters(), lr=learning_rate)

train_loss = train(model=my_unet, num_epochs=num_epochs, batch_size=batch_size,
                   train_set=train_set, optimizer=optimizer)
