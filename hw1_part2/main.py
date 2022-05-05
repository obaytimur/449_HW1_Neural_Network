import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import models
import statistics as stat
from operator import add

torch.manual_seed(70)

# Dataset loader for the different models, isMLP variable needed to change according to used model
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

# general function that trains the model
def train(model, optimizer, loss_criterion, train_dataloader, validation_dataloader, test_dataloader, epochs,
          dev_select, model_name):
    best_accuracy = 0
    best_weight = 0
    loss_avg, val_acc_avg, train_acc_avg = [0] * 120, [0] * 120, [0] * 120

    for run_num in range(10): #running number loop
        model.train()
        for layer in model.children(): #reset the parameters over the runs
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

        train_losses, train_accuracies, validation_accuracies, test_accuracies = [], [], [], []
        for epoch_index in range(epochs): #epoch loop 
            train_loss, train_acc = 0, 0.0
            step_counter = 1
            for images, labels in train_dataloader:
                images = images.to(dev_select)
                labels = labels.to(dev_select)
                optimizer.zero_grad()
                predictions = model(images)
                loss = loss_criterion(predictions, labels)
                loss.backward()			# calculates loss funtion
                optimizer.step()

                train_loss += loss.item()

                predictions = predictions.argmax(dim=1)
                train_acc += (torch.where(predictions == labels)[0].shape[0] / 50)

                if step_counter % 10 == 0: #at each ten steps calculates train loss and validation acc
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
                        f"{model_name}, running number: {run_num:5d}, epoch: {epoch_index:3d}, step: {step_counter:3d}, training acc: {(train_acc / 10):5.2f}, "
                        f"training loss: {(train_loss / 10):6.3f}, validation acc: {val_acc:5.2f}")
                    train_acc = 0.0
                    train_loss = 0

                step_counter += 1
        val_acc_avg = [sum(x) for x in zip(val_acc_avg, validation_accuracies)]  #takes the average of the variables over the epochs
        train_acc_avg = [sum(x) for x in zip(train_acc_avg, train_accuracies)]
        loss_avg = [sum(x) for x in zip(loss_avg, train_losses)]

        torch.save(model, f'entire_{model_name}_running{run_num}')

        test_acc = []	#test loop
        for test_batch, test_labels in test_dataloader:
            test_batch = test_batch.to(dev_select)
            test_labels = test_labels.to(dev_select)
            with torch.no_grad():
                test_predictions = model(test_batch)
                test_predictions = test_predictions.argmax(dim=1)
                test_acc = torch.where(test_predictions == test_labels)
                test_acc = test_acc[0].shape[0] / 50
            test_accuracies.append(test_acc)
        test_accuracies = stat.mean(test_accuracies)
        print(f"Test Accuracy: {test_accuracies}")

        if test_accuracies > best_accuracy:	#if a more succesful test comes saves the data
            best_accuracy = test_accuracies
            torch.save(model.state_dict(), f'model_state_dict_best_accuracy_{model_name}_at_run_{run_num}')
            best_weight = model.fc1.weight.data.numpy().tolist()

    val_acc_avg_new = [i / (run_num + 1) for i in val_acc_avg]	#takes the average of variables over all runs
    train_acc_avg_new = [i / (run_num + 1) for i in train_acc_avg]
    loss_avg_new = [i / (run_num + 1) for i in loss_avg]

    Arch_Dict = {"name": model_name, "loss_curve": loss_avg_new, "train_acc_curve": train_acc_avg_new,
                 "val_acc_curve": val_acc_avg_new, "test_acc": best_accuracy, "weights": best_weight}

    torch.save(Arch_Dict, f'Results_of_the_Architecture_{model_name}') #saves the overall architect


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
model1 = models.mlp_1()
model2 = models.mlp_2()
model3 = models.cnn_3()
model4 = models.cnn_4()
model5 = models.cnn_5()

model1_cuda = model1.to(device)
model2_cuda = model2.to(device)
model3_cuda = model3.to(device)
model4_cuda = model3.to(device)
model5_cuda = model5.to(device)

optimizer1 = torch.optim.Adam(model1_cuda.parameters())
optimizer2 = torch.optim.Adam(model2_cuda.parameters())
optimizer3 = torch.optim.Adam(model3_cuda.parameters())
optimizer4 = torch.optim.Adam(model4_cuda.parameters())
optimizer5 = torch.optim.Adam(model5_cuda.parameters())

# train(model1_cuda, optimizer1, criterion, train_generator, validation_generator, 15, device)
model_1_name = 'Model_1'
model_2_name = 'Model_2'
model_3_name = 'Model_3'
model_4_name = 'Model_4'
model_5_name = 'Model_5'

train(model1_cuda, optimizer1, criterion, train_generator, validation_generator, test_generator, 15, device,
      model_1_name)
train(model2_cuda, optimizer2, criterion, train_generator, validation_generator, test_generator, 15, device,
      model_2_name)
