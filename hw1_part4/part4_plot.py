import utils as u
import torch



dict_1 = torch.load("Results_of_the_Architecture_Model_2_1")
dict_01 = torch.load("Results_of_the_Architecture_Model_2_01")
dict_001 = torch.load("Results_of_the_Architecture_Model_2_001")

dict_list = {"name": 'Model2=MLP2', "loss_curve_1": dict_1['loss_curve'], "loss_curve_01": dict_01['loss_curve'], "loss_curve_001": dict_001['loss_curve'],
                                    "val_acc_curve_1": dict_1['val_acc_curve'], "val_acc_curve_01": dict_01['val_acc_curve'], "val_acc_curve_001": dict_001['val_acc_curve']}

u.part4Plots(dict_list, '', 'plot_part4')