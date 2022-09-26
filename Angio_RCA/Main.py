import json
import pathlib
import numpy as np

import torch
from PIL import Image,ImageShow

import LoadNPZ
from Device import mydevice
from SystemLogs import SystemLogs
from Utils.Visualizer import overlay_imgs
import matplotlib.pyplot as plt

import random
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
from torchvision import transforms
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import ExponentialLR

from Dataset.CenterLineDataset import CenterLineDataset

from Models.SimpleEncoder import SimpleEncoder

# import cv2

from Models.UNet import UNet

with open("Config.json", "r") as f:
    config = json.load(f)


def manual_seed():
    torch.manual_seed(config["optimizer"]["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # print(torch.backends.cudnn.deterministic)
    # print(torch.backends.cudnn.benchmark)
    np.random.seed(config["optimizer"]["seed"])


def main():

    torch.cuda.empty_cache()
    SystemLogs(mydevice)  # print the hostname, pid, device etc
    manual_seed()

    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = LoadNPZ.train_valid_test()
    # list-out Patient-IDs of training data
    print("training: {}".format(len(X_train)))
    print(X_train)

    # list-out Patient-IDs of validation data
    print("validation: {}".format(len(X_valid)))
    print(X_valid)
    # list-out Patient-IDs of test data
    print("test:{}".format(len(X_test)))
    print(X_test)

    annotations = {"train": [], "valid": [], "test": []}
    images = {"train": [], "valid": [], "test": []}
    masks = {"train": [], "valid": [], "test": []}
    central_lines = {"train": [], "valid": [], "test": []}

    for i in range(len(y_train)):
        for j in range(len(y_train[i][1][1])):
            images["train"].append(y_train[i][1][1][str(j)][0])
            masks["train"].append(y_train[i][1][1][str(j)][1])

            coordinates = y_train[i][1][1][str(j)][2]
            annotations["train"].append(coordinates)

            curve = np.zeros((512, 512), dtype=np.uint8)
            for k in range(coordinates.shape[0]):
                curve[coordinates[k][0], coordinates[k][1]] = 255
            central_lines["train"].append(curve)

    for i in range(len(y_valid)):
        images["valid"].append(y_valid[i][1][1]["0"][0])
        masks["valid"].append(y_valid[i][1][1]["0"][1])
        coordinates = y_valid[i][1][1]["0"][2]
        annotations["valid"].append(coordinates)
        # create image with curve
        curve = np.zeros((512, 512), dtype=np.uint8)
        for k in range(coordinates.shape[0]):
            curve[coordinates[k][0], coordinates[k][1]] = 255
        central_lines["valid"].append(curve)

    for i in range(len(y_test)):
        images["test"].append(y_test[i][1][1]["0"][0])
        masks["test"].append(y_test[i][1][1]["0"][1])

        coordinates = y_test[i][1][1]["0"][2]
        annotations["test"].append(coordinates)
        # create image with curve
        curve = np.zeros((512, 512), dtype=np.uint8)
        for k in range(coordinates.shape[0]):
            curve[coordinates[k][0], coordinates[k][1]] = 255
        central_lines["test"].append(curve)

    # check the size
    print(len(images["train"]))
    print(len(images["valid"]))
    print(len(images["test"]))
    #
    print(len(masks["train"]))
    print(len(masks["valid"]))
    print(len(masks["test"]))
    #
    print(len(central_lines["train"]))
    print(len(central_lines["valid"]))
    print(len(central_lines["test"]))
    #
    print(len(annotations["train"]))
    print(len(annotations["valid"]))
    print(len(annotations["test"]))

    """
    # Visually verify images randomly chosen
    # *** add curve into the overlay function ***
    for repeat in range(3):
        n = random.randint(0, len(images["train"]) - 1)
        print("training.. {}".format(n))
        overlay_imgs(images["train"][n], masks["train"][n], annotations["train"][n], n_pts=32).show()
        plt.imshow(masks["train"][n])
        plt.show()
    for repeat in range(3):
        n = random.randint(0, len(images["valid"]) - 1)
        print("valid.. {}".format(n))
        overlay_imgs(images["valid"][n], masks["valid"][n], annotations["valid"][n], n_pts=32).show()
        plt.imshow(masks["valid"][n])
        plt.show()
    for repeat in range(3):
        n = random.randint(0, len(images["test"]) - 1)
        print("test.. {}".format(n))
        overlay_imgs(images["test"][n], masks["test"][n], annotations["test"][n], n_pts=32).show()
        plt.imshow(masks["test"][n])
        plt.show()
    """
    min_max_scaler = MinMaxScaler()
    myCentralLineDataset = {
        "test": CenterLineDataset(images["test"], central_lines["test"], image_transform=min_max_scaler),
        "valid": CenterLineDataset(images["valid"], central_lines["valid"], image_transform=min_max_scaler),
        "train": CenterLineDataset(images["train"], central_lines["train"], image_transform=min_max_scaler)
    }

    myDataLoader = {"test": torch.utils.data.DataLoader(
        myCentralLineDataset["test"], config["model"]["validation_batch_size"], shuffle=False, num_workers=0),
        "valid": torch.utils.data.DataLoader(
            myCentralLineDataset["valid"], config["model"]["validation_batch_size"], shuffle=False, num_workers=0),
        "train": torch.utils.data.DataLoader(
            myCentralLineDataset["train"], config["model"]["train_batch_size"], shuffle=True, num_workers=0)}

    # Model Summary
    # myAutoencoder = SimpleEncoder().to(mydevice)
    # summary(myAutoencoder, input_size=(1, 512, 512), batch_size=10, device='cuda')

    # myEncoder = SimpleEncoder().to(mydevice).double()
    encoder = UNet(1, 1).to(mydevice).double()
    x = torch.randn(1, 1, 512, 512, dtype=torch.double).to(mydevice)
    output = encoder(x)



    # create an optimizer object
    # Adam optimizer with learning rate 1e-3
    optimizer_encoder = optim.Adam(encoder.parameters(), lr=1e-3, weight_decay=1e-7)
    # mean-squared error loss
    criterion = nn.MSELoss()

    # Now start training

    epochs = config["optimizer"]["epoch"]
    loss_dict = {"train_reconstruct": [], "validation_reconstruct": [], "test_reconstruct": []}
    for epoch in range(epochs):

        train_total_batch = 0
        train_running_loss = 0.0

        train_running_loss_reconstruct = 0.0

        tk0 = tqdm(myDataLoader["train"], desc="Iteration")

        encoder.train()
        for step, (batch_input, batch_output) in enumerate(tk0):
            optimizer_encoder.zero_grad()

            batch_input = torch.unsqueeze(batch_input, 1).to(mydevice)
            batch_output = torch.unsqueeze(batch_output, 1).to(mydevice)

            # compute reconstructions
            x = encoder(batch_input)
            loss_reconstruct = torch.sqrt(criterion(x, batch_output))
            loss_reconstruct.backward()
            optimizer_encoder.step()

            train_total_batch += 1
            # add the mini-batch training loss to epoch loss
            train_running_loss_reconstruct += loss_reconstruct.item()

            """
            if epochs == 6 && step == 5:

                h = Image.fromarray(h).convert("RGBA")
                ImageShow.show(h)
            """
        train_loss_reconstruct = train_running_loss_reconstruct / train_total_batch
        loss_dict["train_reconstruct"].append(train_loss_reconstruct)
        print('epoch [%d] training_loss: [%.8f] ' % (epoch, train_loss_reconstruct))

        encoder.eval()
        with torch.no_grad():
            validation_total_batch = 0
            validation_running_loss_reconstruct = 0.0

            i = 0
            for batch_input, batch_output in myDataLoader["valid"]:
                batch_input = torch.unsqueeze(batch_input, 1).to(mydevice)
                batch_output = torch.unsqueeze(batch_output, 1).to(mydevice)

                x = encoder(batch_input)
                loss_reconstruct = torch.sqrt(criterion(x, batch_output))

                validation_running_loss_reconstruct += loss_reconstruct.item()
                # print("validation:"+str(loss_curve_trace.item()))
                validation_total_batch += 1

                if i == 5:
                    h = np.concatenate(((batch_input.squeeze().cpu().data.numpy() * 255).round().astype(np.uint8),
                                    (x.squeeze().cpu().data.numpy() * 255).round().astype(np.uint8),
                                   (batch_output.squeeze().cpu().data.numpy() * 255).round().astype(np.uint8)),axis=1)
                    h = Image.fromarray(h).convert("RGBA")
                    ImageShow.show(h)
                i += 1

            validation_loss_reconstruct = validation_running_loss_reconstruct / validation_total_batch
            loss_dict["validation_reconstruct"].append(validation_loss_reconstruct)
            print('epoch [%d] validation_loss: [%.8f] ' % (epoch, validation_loss_reconstruct))

    return


if __name__ == "__main__":
    main()
