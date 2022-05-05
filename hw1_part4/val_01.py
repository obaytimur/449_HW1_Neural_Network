import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import models
import matplotlib.pyplot as plt
import statistics as stat
from operator import add

torch.manual_seed(70)


class mDataSet(Dataset):
    def __init__(self, data, labels, isMLP=True, is_validation=False):
        self.isMLP = isMLP
        if is_validation:
            validation_indexes = np.random.choice(data.shape[0], int(data.shape[0] / 10), replace=True)
            self.dataset = data[validation_indexes]
            self.labels = labels[validation_indexes]
        else:
            self.dataset = data
            self.labels = labels

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, item_index):
        an_image = self.dataset[item_index]
        an_image = ((an_image / 255) - 0.5) * 2
        a_label = self.labels[item_index]
        if self.isMLP:
            an_image = an_image.reshape(-1)
        else:
            an_image = an_image.unsqueeze(0)

        return an_image, a_label


def train(model, optimizer1, optimizer2, loss_criterion, train_dataloader, validation_dataloader, test_dataloader, epochs,
          dev_select, model_name, opt_name):
    best_accuracy = 0
    best_weight = 0
    loss_avg, val_acc_avg, train_acc_avg = [0] * 120, [0] * 120, [0] * 120

    for run_num in range(1):
        model.train()
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        train_losses, train_accuracies, validation_accuracies, test_accuracies = [], [], [], []
        for epoch_index in range(epochs):
            train_loss, train_acc = 0, 0.0
            step_counter = 1
            for images, labels in train_dataloader:
                if step_counter < 600:
                    optimizer = optimizer1
                else:
                    optimizer = optimizer2
                images = images.to(dev_select)
                labels = labels.to(dev_select)
                optimizer.zero_grad()
                predictions = model(images)
                loss = loss_criterion(predictions, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                predictions = predictions.argmax(dim=1)
                train_acc += (torch.where(predictions == labels)[0].shape[0] / 50)

                if step_counter % 10 == 0:
                    train_losses.append(train_loss / 10)
                    train_accuracies.append(train_acc / 10)
                    val_acc = []
                    for val_batch, val_labels in validation_dataloader:
                        val_batch = val_batch.to(dev_select)
                        val_labels = val_labels.to(dev_select)
                        with torch.no_grad():
                            val_predictions = model(val_batch)
                            val_predictions = val_predictions.argmax(dim=1)
                            val_acc.append(torch.where(val_predictions == val_labels)[0].shape[0] / 50)
                    val_acc = np.asarray(val_acc).mean()
                    validation_accuracies.append(val_acc)

                    print(
                        f"{model_name+'_'+opt_name}, running number: {run_num:5d}, epoch: {epoch_index:3d}, step: {step_counter:3d}, training acc: {(train_acc / 10):5.2f}, "
                        f"training loss: {(train_loss / 10):6.3f}, validation acc: {val_acc:5.2f}")
                    train_acc = 0.0
                    train_loss = 0

                step_counter += 1
        val_acc_avg = [sum(x) for x in zip(val_acc_avg, validation_accuracies)]
        train_acc_avg = [sum(x) for x in zip(train_acc_avg, train_accuracies)]
        loss_avg = [sum(x) for x in zip(loss_avg, train_losses)]

#        torch.save(model, f'entire_{model_name}_running{run_num}')

    val_acc_avg_new = [i / (run_num + 1) for i in val_acc_avg]
    train_acc_avg_new = [i / (run_num + 1) for i in train_acc_avg]
    loss_avg_new = [i / (run_num + 1) for i in loss_avg]

    Arch_Dict = {"name": model_name + '_' + opt_name, "loss_curve": loss_avg_new, "train_acc_curve": train_acc_avg_new,
                 "val_acc_curve": val_acc_avg_new, "test_acc": best_accuracy, "weights": best_weight}

    plt.plot(val_acc_avg_new)
    plt.xlabel("Steps")
    plt.ylabel("Validation Accuracy")
    plt.title("Learning Rate: 0.1 then 0.01")

    plt.show()
    torch.save(Arch_Dict, f'Results_of_the_Architecture_{model_name}_{opt_name}_val_plot_{opt_name}')


#   print(model.state_dict())


#        return train_losses, train_accuracies, validation_accuracies, test_accuracies, first_layer_weight

train_data = torchvision.datasets.FashionMNIST('./data', train=True, download=True, transform=transforms.ToTensor())
test_data = torchvision.datasets.FashionMNIST('./data', train=False, transform=transforms.ToTensor())

training_dataset = mDataSet(train_data.data, train_data.targets, isMLP=True)
validation_dataset = mDataSet(train_data.data, train_data.targets, isMLP=True, is_validation=True)
test_dataset = mDataSet(test_data.data, test_data.targets, isMLP=True)

train_generator = DataLoader(training_dataset, batch_size=50, shuffle=True)
validation_generator = DataLoader(validation_dataset, batch_size=50, shuffle=True)
test_generator = DataLoader(test_dataset, batch_size=50, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.CrossEntropyLoss()

model2 = models.mlp_2()

model2_cuda = model2.to(device)

optimizer2_1 = torch.optim.SGD(model2_cuda.parameters(), 0.1, 0.0)
optimizer2_01 = torch.optim.SGD(model2_cuda.parameters(), 0.01, 0.0)
optimizer2_001 = torch.optim.SGD(model2_cuda.parameters(), 0.001, 0.0)


model_2_name = 'Model_2'


train(model2_cuda, optimizer2_1, optimizer2_01, criterion, train_generator, validation_generator, test_generator, 15, device,
      model_2_name, '1')

