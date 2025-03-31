import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import os
import torch
import torchvision
import seaborn as sns

import parameter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from PIL import Image

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import csv
from network_model import model_net

result_path = "All_Result"


# result_path = "All_Result_BUS"
# result_path = "All_Result_BUS_CDFI"
# result_path = "All_Result_BUS_SE"

class MyDataset(torch.utils.data.Dataset):  # 创建自己的类：MyDataset,这个类是继承的torch.utils.data.Dataset
    def __init__(self, is_train, root):  # 初始化一些需要传入的参数
        super(MyDataset, self).__init__()
        fh = open(root, 'r', newline='', encoding='utf-8')  # 按照传入的路径和txt文本参数，打开这个文本，并读取内容
        fh_reader = csv.reader(fh)
        imgs = []  # 创建一个名为img的空列表，一会儿用来装东西
        for line in fh_reader:  # 按行循环txt文本中的内容
            # print(line)
            imgs.append((line[0], int(line[0][9])))  # Datasets
            # imgs.append((line[0], int(line[0][13])))  # Datasets_BUS
            # imgs.append((line[0], int(line[0][18])))  # Datasets_BUS_CDFI
            # imgs.append((line[0], int(line[0][16])))  # Datasets_BUS_SE
        self.imgs = imgs
        self.is_train = is_train
        if self.is_train:
            self.train_tsf = torchvision.transforms.Compose([
                torchvision.transforms.Resize((224, 224)),
                # torchvision.transforms.RandomResizedCrop(524, scale=(0.1, 1), ratio=(0.5, 2)),
                torchvision.transforms.ToTensor()
            ])
        else:
            self.test_tsf = torchvision.transforms.Compose([
                torchvision.transforms.Resize((224, 224)),
                # torchvision.transforms.CenterCrop(size=500),
                torchvision.transforms.ToTensor()])

    def __getitem__(self, index):  # 这个方法是必须要有的，用于按照索引读取每个元素的具体内容
        feature, label = self.imgs[index]  # fn是图片path #fn和label分别获得imgs[index]也即是刚才每行中word[0]和word[1]的信息
        feature = Image.open(feature).convert('RGB')  # 按照path读入图片from PIL import Image # 按照路径读取图片
        original_width, original_height = feature.size
        crop_rectangle = (30, 220, original_width - 200, original_height - 100)
        feature = feature.crop(crop_rectangle)

        if self.is_train:
            feature = self.train_tsf(feature)
        else:
            feature = self.test_tsf(feature)
        return feature, label

    def __len__(self):  # 这个函数也必须要写，它返回的是数据集的长度，也就是多少张图片，要和loader的长度作区分
        return len(self.imgs)


# 显示混淆矩阵
def plot_confuse_data(true_labels_list, pred_labels_list, model_name, k):
    print(true_labels_list)
    print(pred_labels_list)
    print("Unique labels in true_labels_list:", set(true_labels_list))
    print("Unique labels in pred_labels_list:", set(pred_labels_list))

    display_labels = ["Adenosis", "Cyst", "Papilloma", "Fibroadenoma",
                      "Luminal A", "Luminal B", 'HER2+', 'Triple negative']
    confusion_mat = metrics.confusion_matrix(true_labels_list, pred_labels_list)

    ax = sns.heatmap(confusion_mat,
                     annot=True,
                     fmt='d',
                     cmap='Reds',
                     annot_kws={"fontsize": 12},
                     yticklabels=display_labels,
                     xticklabels=display_labels)

    ax.set_xlabel("Prediction", fontsize=12)
    ax.set_ylabel("Ground truth", fontsize=12)
    # plt.xticks(fontsize=12, rotation=0)
    plt.yticks(fontsize=12, rotation=0)
    plt.savefig(f'{result_path}/result_{model_name}/{model_name}_{k}_confusion-matrix.png', dpi=500,
                bbox_inches='tight')
    plt.close()


def ass_result():
    net_flag = parameter.net_flag

    for k in range(1, 6):

        model, model_name = model_net(net_flag)
        print(model_name)

        dataset_root = f"{result_path}/result_{model_name}/val_{k}.txt"
        weight_root = f"{result_path}/result_{model_name}/{model_name}-k-{k}-best-weight.pt"

        if os.path.exists(f'{result_path}/result_{model_name}/{k}_pred_label_list.txt'):
            true_label_list = []
            pred_label_list = []

            with open(f'{result_path}/result_{model_name}/{k}_pred_label_list.txt', 'r') as file:
                # 逐行读取文件
                for line in file:
                    # 删除行尾的换行符并添加到列表
                    pred_label_list.append(int(line.strip()))

            with open(f'{result_path}/result_{model_name}/{k}_true_label_list.txt', 'r') as file:
                # 逐行读取文件
                for line in file:
                    # 删除行尾的换行符并添加到列表
                    true_label_list.append(int(line.strip()))

        else:
            test_data = MyDataset(is_train=False,
                                  root=dataset_root)

            test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      num_workers=10)

            model_file = weight_root
            model.load_state_dict(torch.load(model_file), strict=True)

            model.cuda()
            model.eval()

            true_label_list = []
            out_scores_list = []
            pred_label_list = []
            with torch.no_grad():
                for X, y in test_loader:
                    model.eval()

                    true_label_list.append(y.item())
                    out_scores_list.append(model(X.to(device)))
                    pred_label_list.append(model(X.to(device)).argmax(dim=1).item())

            print(true_label_list)
            print(pred_label_list)

            with open(f'{result_path}/result_{model_name}/{k}_true_label_list.txt', 'w') as file:
                for item in true_label_list:
                    file.write(f"{item}\n")  # 每个元素写入一行

            with open(f'{result_path}/result_{model_name}/{k}_pred_label_list.txt', 'w') as file:
                for item in pred_label_list:
                    file.write(f"{item}\n")  # 每个元素写入一行

        plot_confuse_data(true_label_list, pred_label_list, model_name, k)

        acc_score = []
        specificity = []
        conf_matrix = metrics.confusion_matrix(true_label_list, pred_label_list)
        num_classes = conf_matrix.shape[0]

        acc_score_all = metrics.accuracy_score(true_label_list, pred_label_list)
        precision_score_all = metrics.precision_score(true_label_list, pred_label_list, average='macro')
        recall_score_all = metrics.recall_score(true_label_list, pred_label_list, average='macro')

        def calculate_specificity(cm):
            total_specificity = 0
            num_classes = cm.shape[0]
            for i in range(num_classes):
                tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))  # True Negatives
                fp = np.sum(np.delete(cm, i, axis=0)[:, i])  # False Positives
                specificity = tn / (tn + fp)
                total_specificity += specificity
            return total_specificity / num_classes

        specificity_all = calculate_specificity(conf_matrix)

        f1score_all = metrics.f1_score(true_label_list, pred_label_list, average='macro')

        class_accuracies = conf_matrix.diagonal() / conf_matrix.sum(axis=1)
        for idx, acc in enumerate(class_accuracies):
            acc_score.append(acc)

        acc_score = [x * 100 for x in acc_score]
        precision_score = metrics.precision_score(true_label_list, pred_label_list, average=None)
        recall_score = metrics.recall_score(true_label_list, pred_label_list, average=None)
        specificities = np.zeros(num_classes)

        for i in range(num_classes):
            tn = np.sum(conf_matrix) - (np.sum(conf_matrix[i, :]) + np.sum(conf_matrix[:, i]) - conf_matrix[i, i])
            fp = np.sum(conf_matrix[:, i]) - conf_matrix[i, i]
            specificities[i] = tn / (tn + fp)

        specificities = [x * 100 for x in specificities]
        for idx, spec in enumerate(specificities):
            specificity.append(spec)

        f1score = metrics.f1_score(true_label_list, pred_label_list, average=None)

        kappa_all = metrics.cohen_kappa_score(true_label_list, pred_label_list)

        print("总Accuracy(%):" + str(acc_score_all * 100))
        print("总Precision(%):" + str(precision_score_all * 100))
        print("总Recall(%):" + str(recall_score_all * 100))
        print("总Specificity(%):" + str(specificity_all * 100))
        print("总F1-score:" + str(f1score_all))
        print("总Kappa:" + str(kappa_all))

        print("每类Accuracy(%):" + str(acc_score))
        print("每类Precision(%):" + str(precision_score * 100))
        print("每类Recall(%):" + str(recall_score * 100))
        print("每类Specificity(%):" + str(specificity))
        print("每类f1-score:" + str(f1score))

        f = open(f"{result_path}/result_{model_name}/{model_name}_{k}_results.txt", "w")
        f.write("\n总Accuracy(%):" + str(acc_score_all * 100))
        f.write("\n总Precision(%):" + str(precision_score_all * 100))
        f.write("\n总Recall(%):" + str(recall_score_all * 100))
        f.write("\n总Specificity(%):" + str(specificity_all * 100))
        f.write("\n总f1-score:" + str(f1score_all))
        f.write("\n总Kappa:" + str(kappa_all))

        f.write("\n每类Accuracy(%):" + str(acc_score))
        f.write("\n每类Precision(%):" + str(precision_score * 100))
        f.write("\n每类Recall(%):" + str(recall_score * 100))
        f.write("\n每类Specificity(%):" + str(specificity))
        f.write("\n每类f1-score:" + str(f1score))


if __name__ == '__main__':
    ass_result()
