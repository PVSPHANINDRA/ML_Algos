from sys import argv as args
import pandas as pd
import numpy as np
from decisionTree import execute as dt_execute
from nn import execute as nn_execute
from randomForest import execute as rf_execute

csv_contents = pd.read_csv('loan.csv', header=None)
instances = csv_contents.values
# removing the headers
instances = instances[1:]
np_instances = np.array(instances)
np_instances = np_instances[:, 1:]
instances = np_instances.tolist()

class_index = len(instances[0]) - 1

num_attr = [5, 6, 7, 8]
print(instances[0])
print('instances count: ', len(instances))

# if len(args) != 2:
#     print('Please file path for dataset and pass criterion : gini or ig')
# else:
#     criterion = args[1]
#     if criterion != 'gini' and criterion != 'ig':
#         print('Please pass criterion : gini or ig')
#     execute('Contraceptive', instances, class_index, num_attr, criterion)


rf_execute('Loan', instances, class_index, num_attr, 'ig')
# dt_execute('Loan', instances, class_index, num_attr, 'ig')


#hidden_layer = [32]
#_lambda = 0.01
#alpha = 0.4
#max_itr = 200

# instances - 480
#acc, f1_score, reg_costs, err_costs, itr = nn_execute("Loan", instances, class_index, num_attr, hidden_layer,
 #                                                     _lambda, alpha, max_itr, min_grad_diff=0.000001,
  #                                                    mini_batch=len(instances) / 2)
