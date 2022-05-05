import utils as u
import torch

model_name_array = ["Model_1", "Model_2", "Model_3", "Model_4", "Model_5"]
dict_list = []

for model_name in model_name_array:
    file_path_relu = "Results_of_the_Architecture_" + model_name
    file_path_sig = "Results_of_the_Architecture_" + model_name + "_sig"
    dict_1 = torch.load(file_path_relu)
    dict_2 = torch.load(file_path_sig)
    dict_list.append({"name": model_name, "relu_loss_curve": dict_1['loss_curve'], "sigmoid_loss_curve": dict_2['loss_curve'],
                      "relu_grad_curve": dict_1['grad_curve'], "sigmoid_grad_curve": dict_2['grad_curve']})

u.part3Plots(dict_list, '', 'part3_plot')