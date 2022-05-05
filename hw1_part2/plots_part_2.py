import utils as u
import torch

model_name_array = ["Model_1", "Model_2", "Model_3", "Model_4", "Model_5"]
data = []
weights = []
for data_name in model_name_array:
    file_path = "Results_of_the_Architecture_"+ data_name
    data.append(torch.load(file_path))

for weight_name in model_name_array:
    file_path_weight = "best_model/model_state_dict_best_accuracy_" + weight_name
    weights.append(torch.load(file_path_weight,map_location=torch.device('cpu')))
weights_new = []
for i in range(len(weights)):
    if i < 2:
        weights_new.append(weights[i]['fc1.weight'])
    else:
        weights_new.append(weights[i]['conv1.weight'])

u.part2Plots(data)
for i in range(5):
    u.visualizeWeights(weights_new[i], '', f'{model_name_array[i]}')




