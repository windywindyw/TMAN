import os
import numpy as np
import torch
import torchvision
import shutil
import random
import csv
import matplotlib.pyplot as plt
from PIL import Image
from torch import optim
from network_model import model_net
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset, Subset

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
device = torch.device('cuda')


def save_paths(paths, file_name):
    with open(file_name, 'w') as file:
        for path in paths:
            file.write(path + '\n')


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, root):
        super(MyDataset, self).__init__()
        fh = open(root, 'r', newline='')
        fh_reader = csv.reader(fh)
        imgs = []
        for line in fh_reader:
            imgs.append((line[0], int(line[1])))
        self.imgs = imgs
        self.tsf = torchvision.transforms.Compose([
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.ToTensor()
        ])

    def __getitem__(self, index):
        feature, label = self.imgs[index]
        feature = Image.open(feature).convert('RGB')
        original_width, original_height = feature.size
        crop_rectangle = (30, 220, original_width - 200, original_height - 100)
        feature = feature.crop(crop_rectangle)
        feature = self.tsf(feature)
        return feature, label

    def __len__(self):
        return len(self.imgs)

    def get_image_paths(self):
        return [img[0] for img in self.imgs]


def train_main(flag, random_state=None):
    initial_lr = 1e-4
    min_lr = 1e-4
    wd = 1e-4
    batch_size = 8
    num_epochs = 150
    patience = 15

    _, model_name = model_net(flag)
    data_root = "Datasets.txt"
    # data_root = "Datasets_BUS.txt"
    # data_root = "Datasets_BUS_CDFI.txt"
    # data_root = "Datasets_BUS_SE.txt"

    dataset = MyDataset(root=data_root)

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    if os.path.exists("result_" + model_name):
        shutil.rmtree("result_" + model_name)
    os.mkdir("result_" + model_name)
    print(f"{model_name} 开始训练！")
    val_acc_results_all = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        best_epoch = 0
        patience_counter = 0
        best_val_acc = 0
        current_epoch = 0
        min_val_loss = 99
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=10)
        val_loader = DataLoader(val_subset, batch_size=batch_size, num_workers=10)

        model, model_name = model_net(flag)
        model.to(device)
        optimizer = optim.AdamW(model.parameters(), lr=initial_lr, weight_decay=wd)
        scheduler = CosineAnnealingLR(optimizer, T_max=200, eta_min=min_lr)

        loss = torch.nn.CrossEntropyLoss()


        # Saving train and val image paths
        train_image_paths = [dataset.get_image_paths()[i] for i in train_idx]
        val_image_paths = [dataset.get_image_paths()[i] for i in val_idx]

        save_paths(train_image_paths, "result_" + model_name + "/train_" + str(fold+1) + ".txt")
        save_paths(val_image_paths, "result_" + model_name + "/val_" + str(fold+1) + ".txt")

        test_acc_max_l = []

        # Lists to store loss and accuracy values for plotting
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(num_epochs):
            model.train()
            train_total_loss = 0.0
            train_total_acc = 0

            for images, masks in train_loader:
                images, masks = images.to(device), masks.to(device)

                optimizer.zero_grad()
                y_hat = model(images)
                loss_v = loss(y_hat, masks)
                loss_v.backward()
                optimizer.step()

                train_total_loss += loss_v.cpu().item()
                train_total_acc += (y_hat.argmax(dim=1) == masks).sum().cpu().item()

            scheduler.step()
            # 至此，每个epoches完成
            print(f"Epoch {epoch + 1}/{num_epochs}, LR: {scheduler.get_last_lr()}")

            avg_train_loss = train_total_loss / len(train_loader.dataset)
            avg_train_acc = train_total_acc / len(train_loader.dataset)

            print(
                f"Fold {fold}, "
                f"Epoch {epoch}, " +
                "\n"+
                f"Train Loss: {avg_train_loss:.5f}, "
                f"Train Acc: {avg_train_acc:.4f}, "
            )

            train_losses.append(avg_train_loss)
            train_accuracies.append(avg_train_acc)

            model.eval()
            val_total_loss = 0.0
            val_total_acc = 0

            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(device), masks.to(device)

                    y_hat = model(images)
                    loss_v = loss(y_hat, masks)
                    val_total_loss += loss_v
                    val_total_acc += (y_hat.argmax(dim=1) == masks).sum().cpu().item()

                avg_val_loss = val_total_loss / len(val_loader.dataset)
                avg_val_acc = val_total_acc / len(val_loader.dataset)

            val_losses.append(avg_val_loss.cpu().item())
            val_accuracies.append(avg_val_acc)

            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                best_model_wts = model.state_dict()
                best_epoch = epoch
                torch.save(best_model_wts,
                           "result_" + model_name + "/" + model_name + "-k-" + str(fold + 1) + "-best-weight.pt")

            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1


            print(
                f"Val Loss: {avg_val_loss:.5f}, "
                f"Val Acc: {avg_val_acc:.4f}, "
                + "\n"
            )

            test_acc_max_l.append(avg_val_acc)

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                current_epoch = epoch+1
                break  # Early stopping

        index_max = test_acc_max_l.index(max(test_acc_max_l))

        f = open("result_" + model_name + "/results.txt", "a")
        f.write('\n' + "fold" + str(fold) + ":" + "\n" +
                "Acc : " + str(test_acc_max_l[index_max]) + "\n"
                "Best Epoch:" + str(best_epoch+1) + "\n"
                )
        f.close()

        val_acc_results_all.append(test_acc_max_l[index_max])

        # Plotting and saving loss and accuracy curves
        epochs_range = range(1, current_epoch+1)
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, train_losses, label='Train Loss')
        plt.plot(epochs_range, val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title(f'Fold {fold + 1} Loss')
        plt.legend()
        # plt.savefig(f"result_{model_name}/fold_{fold + 1}_loss_curve.png")

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
        plt.plot(epochs_range, val_accuracies, label='Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title(f'Fold {fold + 1} Accuracy')
        plt.legend()
        plt.savefig(f"result_{model_name}/fold_{fold + 1}_accuracy_curve.png")

        plt.close()

    f = open("result_" + model_name + "/results.txt", "a")
    f.write(
        # "Datasets:" + data_root + "\n" +
        "\n"
        "Average Result:" + "\n" +
        "Average Acc:" + str(np.mean(val_acc_results_all)) + "\n" 

    )
    print(
        "Average Result:" + "\n" +
        "Average Acc:" + str(np.mean(val_acc_results_all)) + "\n"
    )

    f.write(
        "\n"
        "Std Result:" + "\n"
        "Std Acc:" + str(np.std(val_acc_results_all)) + "\n"
    )
    print(
        "Std Result:" + "\n"
        "Std Acc:" + str(np.std(val_acc_results_all)) + "\n"
    )


if __name__ == '__main__':
    import time
    start = time.time()

    for i in [17]:
        random_state = random.randint(0, 10000)
        train_main(i, random_state=random_state)
        elapsed_time = time.time() - start
        elapsed_time_minutes = elapsed_time / 60
        # print("训练时间：" + str(elapsed_time) + " 秒")
        print("训练时间：" + str(elapsed_time_minutes) + " 分钟")


