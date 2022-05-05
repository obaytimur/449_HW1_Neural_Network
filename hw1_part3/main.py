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


class mDataSet(Dataset): #same as part2, loads dataset for the right model
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


def train(model, optimizer, loss_criterion, train_dataloader, epochs, dev_select, model_name, is_cnn): #training function

    loss_avg = [0] * 120
    grad_losses = []
    grad_mag = []
    for run_num in range(1):
        model.train()
        for layer in model.children(): #reset the parameters over the runs 
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        train_losses = []
        for epoch_index in range(epochs):
            train_loss, grad_loss = 0, 0
            step_counter = 1
            for images, labels in train_dataloader:
                images = images.to(dev_select)
                labels = labels.to(dev_select)
                optimizer.zero_grad()
                predictions = model(images)
                loss = loss_criterion(predictions, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                if step_counter % 10 == 0:
                    train_losses.append(train_loss / 10)
                    print(
                        f"{model_name}, running number: {run_num:5d}, epoch: {epoch_index:3d}, step: {step_counter:3d}, training loss: {(train_loss / 10):6.3f}")
                    train_loss = 0
                    if is_cnn: # saves the weights of the first layer to calculate gradient magnitude
                        grad_loss_array = model.conv1.weight[0].data.numpy()
                        grad_losses.append(grad_loss_array.copy())
                    else:
                        grad_loss_array = model.fc1.weight[0].data.numpy()
                        grad_losses.append(grad_loss_array.copy())

                step_counter += 1

        loss_avg = [sum(x) for x in zip(loss_avg, train_losses)]
#        grad_avg = [sum(x) for x in zip(grad_avg, grad_losses)]

    loss_avg_new = [i / (run_num + 1) for i in loss_avg]
#    grad_avg_new = [i / (run_num + 1) for i in grad_avg]

    for i in range(len(grad_losses) - 1): #loop that calculates grad. magnitudes
        grad_mag.append(np.linalg.norm(grad_losses[i] - grad_losses[i+1]) / 0.01)


    Arch_Dict = {"name": model_name, "loss_curve": loss_avg_new, "grad_curve": grad_mag}

    torch.save(Arch_Dict, f'Results_of_the_Architecture_{model_name}')


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
model1_sig = models.mlp_1_sig()
model2 = models.mlp_2()
model2_sig = models.mlp_2_sig()
model3 = models.cnn_3()
model3_sig = models.cnn_3_sig()
model4 = models.cnn_4()
model4_sig = models.cnn_4_sig()
model5 = models.cnn_5()
model5_sig = models.cnn_5_sig()

model1_cuda = model1.to(device)
model1_sig_cuda = model1_sig.to(device)
model2_cuda = model2.to(device)
model2_sig_cuda = model2_sig.to(device)
model3_cuda = model3.to(device)
model3_sig_cuda = model3_sig.to(device)
model4_cuda = model3.to(device)
model4_sig_cuda = model4_sig.to(device)
model5_cuda = model5.to(device)
model5_sig_cuda = model5_sig.to(device)

optimizer1 = torch.optim.SGD(model1_cuda.parameters(), lr=0.01, momentum=0.0)
optimizer1_sig = torch.optim.SGD(model1_sig_cuda.parameters(), lr=0.01, momentum=0.0)
optimizer2 = torch.optim.SGD(model2_cuda.parameters(), lr=0.01, momentum=0.0)
optimizer2_sig = torch.optim.SGD(model2_sig_cuda.parameters(), lr=0.01, momentum=0.0)
optimizer3 = torch.optim.SGD(model3_cuda.parameters(), 0.01, 0.0)
optimizer3_sig = torch.optim.SGD(model3_sig_cuda.parameters(), 0.01, 0.0)
optimizer4 = torch.optim.SGD(model4_cuda.parameters(), 0.01, 0.0)
optimizer4_sig = torch.optim.SGD(model4_sig_cuda.parameters(), 0.01, 0.0)
optimizer5 = torch.optim.SGD(model5_cuda.parameters(), 0.01, 0.0)
optimizer5_sig = torch.optim.SGD(model5_sig_cuda.parameters(), 0.01, 0.0)

model_1_name = 'Model_1'
model_1_sig_name = 'Model_1_sig'
model_2_name = 'Model_2'
model_2_sig_name = 'Model_2_sig'
model_3_name = 'Model_3'
model_3_sig_name = 'Model_3_sig'
model_4_name = 'Model_4'
model_4_sig_name = 'Model_4_sig'
model_5_name = 'Model_5'
model_5_sig_name = 'Model_5_sig'

train(model1_cuda, optimizer1, criterion, train_generator, 15, device, model_1_name, False)
train(model1_sig_cuda, optimizer1_sig, criterion, train_generator, 15, device, model_1_sig_name, False)
train(model2_cuda, optimizer2, criterion, train_generator, 15, device, model_2_name, False)
train(model2_sig_cuda, optimizer2_sig, criterion, train_generator, 15, device, model_2_sig_name, False)


#train(model3_cuda, optimizer3, criterion, train_generator, 15, device, model_3_name, True)
#train(model3_sig_cuda, optimizer3_sig, criterion, train_generator, 15, device, model_3_sig_name, True)
#train(model4_cuda, optimizer4, criterion, train_generator, 15, device, model_4_name, True)
#train(model4_sig_cuda, optimizer4_sig, criterion, train_generator, 15, device, model_4_sig_name, True)
#train(model5_cuda, optimizer5, criterion, train_generator, 15, device, model_5_name, True)
#train(model5_sig_cuda, optimizer5_sig, criterion, train_generator, 15, device, model_5_sig_name, True)
