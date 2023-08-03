import sklearn
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plot

k = 10


def sigmoid(z):
    return 1 / (1 + np.exp(-1 * z))


def get_z(theta, inputs):
    return np.dot(theta, inputs.transpose()).transpose()


def get_activation(z):
    return sigmoid(z)


def get_error_cost(predicted_op, expected_op):
    cost = np.multiply((-1 * expected_op), np.log(predicted_op)) - np.multiply((1 - expected_op), np.log(1 - predicted_op))
    return np.sum(cost)


def get_weights_sq_sum(weights):
    sq_sum = 0
    for weight in weights:
        removed_bias_weights = weight[:, 1:]
        squared = np.multiply(removed_bias_weights, removed_bias_weights)
        sq_sum += np.sum(squared)
    return sq_sum


# return array<ndarray<float>>
# get delta for each neuron for the layer
# first layer which is input --> for this delta will be zero
# other layer delta will be ndarray<float> where size = next_layer_neurons * 1
def get_delta(activations, weights, expected_op):
    total_layers = len(activations)
    deltas = [None] * total_layers
    for layer in range(total_layers - 1, 0, -1):
        # final layer
        if layer == total_layers - 1:
            deltas[layer] = np.subtract(activations[layer], expected_op)
            continue
        # removing the bias
        theta = weights[layer][:, 1:]
        activation = activations[layer][:, 1:]
        next_layer_delta = deltas[layer + 1]
        diff = np.dot(theta.transpose(), next_layer_delta.transpose()).transpose()
        deltas[layer] = np.multiply(diff, np.multiply(activation, 1 - activation)).reshape(1, -1)

    # print("deltas", deltas)
    return deltas


def get_gradients(activations, deltas):
    total_layers = len(activations)
    gradients = [None] * (total_layers - 1)
    for layer in range(0, total_layers - 1):
        activation = activations[layer]
        next_layer_delta = deltas[layer + 1]
        gradients[layer] = np.dot(next_layer_delta.transpose(), activation)
    # print(gradients)
    return gradients


def get_gradient_sum(g1, g2):
    result = []
    for i in range(len(g1)):
        result.append(g1[i] + g2[i])
    return result


# takes input and weights
# iterate on each layer and finds out the activation of next layer
# returns the array<np.array<activation>>
def get_activations(input_inst, weights):
    activation = input_inst
    z_value = None
    z_values = []
    activations = []
    for theta in weights:
        activation = np.append([[1]], activation.reshape(1, -1), axis=1)
        z_values.append(z_value)
        activations.append(activation)
        z_value = get_z(theta, activation)
        activation = get_activation(z_value)
    activations.append(activation.reshape(1, -1))
    z_values.append(z_value)
    return z_values, activations


# returns the tuple(val1, val2)
# val1 --> activations (array of activation of each layer)
# val2 --> error cost of the instance
def forward_prop(input_inst, weights, est_op):
    z_values, activations = get_activations(input_inst, weights)
    activation = activations[-1]
    err_cost = get_error_cost(activation, est_op)
    # print('cost', err_cost)
    return activations, err_cost


# trying to find the
def backward_prop(activations, weights, expected_op):
    deltas = get_delta(activations, weights, expected_op)
    gradients = get_gradients(activations, deltas)
    return gradients


def avg_reg_cost(err_cost_sum, _lambda, weights, n):
    penalising_weight_sum = get_weights_sq_sum(weights)
    return (err_cost_sum / n) + (_lambda / (2 * n)) * penalising_weight_sum


def avg_reg_gradient(gradient_sum, _lambda, weights, n):
    result = []
    for i in range(len(gradient_sum)):
        gradient_sum[i][:, 1:] += _lambda * weights[i][:, 1:]
        result.append(1 / n * gradient_sum[i])
    return result


def update_weights(weights, reg_gradient, alpha):
    result = []
    for i in range(len(reg_gradient)):
        result.append(np.subtract(weights[i], alpha * reg_gradient[i]))
    return result


def algo(inputs, outputs, weights, _lambda, logging=False):
    err_cost_sum = 0
    gradient_sum = [0 * weight for weight in weights]

    for _i in range(len(inputs)):
        instance_x = inputs[_i]
        instance_y = outputs[_i]

        # forward propagation
        z_values, activations = get_activations(instance_x, weights)
        activation = activations[-1]
        err_cost = get_error_cost(activation, instance_y)

        if logging:
            print("Processing training instance {}".format(_i + 1))
            print("Forward propagating the input ", np.round(instance_x, 5).tolist())
            for i in range(len(z_values)):
                z_value = z_values[i]
                activation = activations[i]
                if z_value is not None:
                    print("z{}: {}".format(i + 1, np.round(z_value, 5).tolist()))
                print("a{}: {}".format(i + 1, np.round(activation, 5).tolist()))
                print("\n")

            print("f(x): ", activations[-1].tolist())
            print("Predicted output for instance {}: {}".format(_i + 1, np.round(activations[-1], 5).tolist()))
            print("Expected output for instance {}: {}".format(_i + 1, np.round(instance_y, 5).tolist()))
            print("Cost, J, associated with instance {}: {}".format(_i + 1, round(err_cost, 5)))
            print("\n")

        # backward propagation
        deltas = get_delta(activations, weights, instance_y)
        gradients = get_gradients(activations, deltas)

        if logging:
            print("Running backpropagation:")
            print("Computing gradients based on training instance {}".format(_i + 1))
            for i in range(len(deltas) - 1, 0, -1):
                delta = deltas[i]
                print("delta{}: {}".format(i + 1, np.round(delta, 5).tolist()))

            print("\n")

            for i in range(len(gradients) - 1, -1, -1):
                gradient = gradients[i]
                print("Gradients of Theta{} based on training instance {}:".format(i + 1, _i + 1))
                print(np.round(gradient, 5).tolist())
                print("\n")
            print("###############################")
            print("\n")

        # adding to aggregator
        err_cost_sum += err_cost
        gradient_sum = get_gradient_sum(gradient_sum, gradients)

    # print("error_cost_sum", err_cost_sum)

    # printing aggregator variable values
    reg_cost = avg_reg_cost(err_cost_sum, _lambda, weights, len(inputs))
    final_reg_gradients = avg_reg_gradient(gradient_sum, _lambda, weights, len(inputs))

    if logging:
        print("Final (regularized) cost, J, based on the complete training set: {}".format(round(reg_cost, 5)))
        print("\n")
        print("The entire training set has been processed. Computing the average (regularized) gradients:")

        for _i in range(len(final_reg_gradients)):
            final_reg_gradient = final_reg_gradients[_i]
            print("Final regularized gradients of Theta{}:".format(_i + 1))
            print(np.round(final_reg_gradient, 5).tolist())
            print("\n")
    return err_cost_sum, reg_cost, final_reg_gradients


class NeuralNetwork:
    def __init__(self, training_set, class_index, num_attr, hidden_layer_neurons, _lambda, alpha, max_itr, min_grad_diff=float("-inf"),
                 mini_batch=None):
        self.class_index = class_index
        self._lambda = _lambda
        self.num_attr = num_attr
        self.alpha = alpha
        self.itr = 0
        self.reg_cost = 0
        self.err_cost = 0
        self.max_itr = max_itr
        self.min_grad_diff = min_grad_diff
        self.mini_batch = mini_batch if mini_batch is not None else len(training_set)

        np_training_set = np.array(training_set)
        self.np_training_set = np_training_set
        self.class_labels = None
        self.norm_np_train_set = self.apply_normalize_n_one_hot_encoding(np_training_set)
        outputs_count = len(self.class_labels)
        inputs_count = len(self.norm_np_train_set[0]) - outputs_count
        neurons = [inputs_count]
        neurons.extend(hidden_layer_neurons)
        neurons.append(outputs_count)
        self.neurons = neurons
        self.weights = self.get_rand_weights()
        self.inputs_count = inputs_count
        self.outputs_count = outputs_count
        self.build()

    def apply_one_hot_encoding(self, np_dataset, column_index):
        np_column_values = np_dataset[:, column_index].reshape(-1, 1)
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(self.np_training_set[:, column_index].reshape(-1, 1))
        if column_index == self.class_index and self.class_labels is None:
            self.class_labels = enc.categories_[0].tolist()
        return enc.transform(np_column_values).toarray()

    # apply one hot encoding n normalize:
    def apply_normalize_n_one_hot_encoding(self, np_dataset):
        # placing class column in the start
        inputs = self.apply_one_hot_encoding(np_dataset, self.class_index)

        for column_index in range(len(np_dataset[0])):
            if column_index == self.class_index:
                continue
            if column_index in self.num_attr:
                self.np_training_set[:, column_index] = self.np_training_set[:, column_index].astype(float)
                np_dataset[:, column_index] = np_dataset[:, column_index].astype(float)
                min_value = np.min(self.np_training_set[:, column_index].astype(float))
                max_value = np.max(self.np_training_set[:, column_index].astype(float))
                column_values = np_dataset[:, column_index].astype(float).reshape(-1, 1)
                diff = (max_value - min_value)
                norm_column_values = (column_values - min_value) / diff if diff > 0 else column_values
                inputs = np.append(inputs, norm_column_values, axis=1)
            else:
                inputs = np.append(inputs, self.apply_one_hot_encoding(np_dataset, column_index), axis=1)
        return inputs.astype(float)

    # getting theta for each layer
    def get_rand_weights(self):
        thetas = []
        for index in range(len(self.neurons) - 1):
            theta = np.random.randn(self.neurons[index + 1], self.neurons[index] + 1)
            thetas.append(theta)
        return thetas

    def build(self):
        inputs = self.norm_np_train_set[:, self.outputs_count:]
        outputs = self.norm_np_train_set[:, :self.outputs_count]
        start = 0
        while True:
            end = start + self.mini_batch
            if end > self.mini_batch:
                end = self.mini_batch

            err_cost_sum, reg_cost, reg_gradients = algo(inputs[start:end], outputs[start:end], self.weights, self._lambda)
            # print("err_cost_sum", err_cost_sum)
            # print("reg_cost", reg_cost)

            # update weights
            self.weights = update_weights(self.weights, reg_gradients, self.alpha)
            self.err_cost = err_cost_sum / (end - start)
            if end == self.mini_batch:
                self.itr += 1
                start = 0
            else:
                start = end

            if self.reg_cost == 0:
                self.reg_cost = reg_cost
                continue

            grad_diff = self.reg_cost - reg_cost
            self.reg_cost = reg_cost
            # print("grad_diff", grad_diff)
            print("reg_cost", reg_cost)

            if abs(grad_diff) < self.min_grad_diff or self.itr >= self.max_itr:
                print(self.itr)
                break
        # print(self.weights)
        # print(self.reg_cost)

    def get_predicted_op_index(self, np_input_inst):
        z_values, act = get_activations(np_input_inst, self.weights)
        prob = act[-1]
        return np.argmax(prob)

    # returns confusion matrix
    def get_confusion_matrix(self, testing_set):
        np_testing_set = np.array(testing_set)
        norm_test_set = self.apply_normalize_n_one_hot_encoding(np_testing_set)
        inputs = norm_test_set[:, self.outputs_count:]
        outputs = norm_test_set[:, :self.outputs_count]
        # i*j --> i is inst label and j is predicted label
        matrix = [[0 for _ in self.class_labels] for _ in self.class_labels]

        for _index in range(len(testing_set)):
            predicted_op = self.get_predicted_op_index(inputs[_index])
            exp_op = np.argmax(outputs[_index])
            matrix[exp_op][predicted_op] += 1
        return matrix

    def get_cost(self, dataset):
        np_testing_set = np.array(dataset)
        norm_test_set = self.apply_normalize_n_one_hot_encoding(np_testing_set)
        inputs = norm_test_set[:, self.outputs_count:]
        outputs = norm_test_set[:, :self.outputs_count]
        err_costs = []
        for _index in range(len(dataset)):
            z_values, activations = get_activations(inputs[_index], self.weights)
            activation = activations[-1]
            err_cost = get_error_cost(activation, outputs[_index])
            err_costs.append(err_cost)
        return get_mean(err_costs)


# returns folds
def create_folds(instances, class_index):
    # shuffle
    instances = sklearn.utils.shuffle(instances)
    # map with key:class_label and value: instances
    class_map = {}
    for inst in instances:
        class_value = inst[class_index]
        if class_value not in class_map:
            class_map[class_value] = [inst]
        else:
            class_map[class_value].append(inst)

    # map with key:class_label and value: start(from where instances have to copy)
    index_map = {}
    for label in class_map:
        index_map[label] = 0

    folds = []
    for i in range(0, k):
        fold = []
        for label in index_map:
            start = index_map[label]
            total = len(class_map[label])
            end = start + round(total / k)
            # 10/4 --> end 2,2,2,2 --> missing last two instances.
            if i == k - 1:
                end = total
            fold.extend(class_map[label][start:end])
            index_map[label] = end
        folds.append(fold)
    return folds


def execute(dataset_name, dataset, class_index, num_attr, hidden_layer_neurons, _lambda, alpha, max_itr, min_grad_diff=float("-inf"),
            mini_batch=None):
    print("Dataset name:", dataset_name)
    print("execute")
    folds = create_folds(dataset, class_index)
    acc = []
    precision = []
    recall = []
    f1_score = []
    reg_costs = []
    iterations = []
    err_costs = []
    for testing_index in range(k):
        testing_set = folds[testing_index]
        training_set = []
        for index in range(len(folds)):
            if index != testing_index:
                training_set.extend(folds[index])
        nn = NeuralNetwork(training_set, class_index, num_attr, hidden_layer_neurons, _lambda, alpha, max_itr, min_grad_diff,
                           mini_batch)
        # matrix = nn.get_confusion_matrix(training_set)
        matrix = nn.get_confusion_matrix(testing_set)
        precision_value = get_precision(matrix)
        recall_value = get_recall(matrix)
        acc.append(get_accuracy(matrix))
        recall.append(recall_value)
        precision.append(precision_value)
        f1_score.append(get_f1_score(precision_value, recall_value))
        reg_costs.append(nn.reg_cost)
        iterations.append(nn.itr)
        err_costs.append(nn.err_cost)
        # print(testing_index, nn.weights)
        # if testing_index == 1:
        #     break

    print("accuracies:", acc)
    print("f1 score:", f1_score)
    print("reg_costs:", reg_costs)
    print("err_costs:", err_costs)
    print(get_mean(acc))
    print(get_mean(f1_score))
    return get_mean(acc), get_mean(f1_score), get_mean(reg_costs), get_mean(err_costs), get_mean(iterations)


def plot_J(dataset, class_index, num_attr, hidden_layer_neurons, _lambda, alpha, max_itr, min_grad_diff=float("-inf"),
              mini_batch=None):
    folds = create_folds(dataset, class_index)
    err_costs = []
    for testing_index in range(k):
        testing_set = folds[testing_index]
        training_set = []
        for index in range(len(folds)):
            if index != testing_index:
                training_set.extend(folds[index])

        partitioned_set = create_folds(training_set, class_index)
        pointer = 0
        err_cost = []
        while pointer != 10:
            temp_set = partitioned_set[0: pointer + 1]
            train_set = []
            for temp in temp_set:
                train_set.extend(temp)
            train_set = sklearn.utils.shuffle(train_set)
            nn = NeuralNetwork(train_set, class_index, num_attr, hidden_layer_neurons, _lambda, alpha, max_itr, min_grad_diff)
            err_cost.append(nn.get_cost(testing_set))
            pointer += 1
        err_costs.append(err_cost)
        # if testing_index == 1:
        #     break
    x = [10 * i for i in range(1, 11)]
    print(err_costs)
    y = np.mean(err_costs, axis=0).reshape(1, -1).tolist()
    plot.plot(x, y[0])
    plot.xlabel('sample percentage')
    plot.ylabel('Cost Function J')
    plot.scatter(x, y[0])
    plot.show()


def get_accuracy(matrix):
    num = 0
    den = 0
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            # true positives
            if i == j:
                num += matrix[i][j]
            den += matrix[i][j]

    return num / den


def get_recall(matrix):
    i = 0
    recall_sum = 0
    while i < len(matrix):
        num = matrix[i][i]
        den = 0
        for j in range(len(matrix)):
            den += matrix[i][j]
        recall = 0 if num == 0 else num / den
        recall_sum += recall
        i += 1
    return round(recall_sum / len(matrix), 5)


def get_precision(matrix):
    i = 0
    precision_sum = 0
    while i < len(matrix):
        num = matrix[i][i]
        den = 0
        for j in range(len(matrix)):
            den += matrix[j][i]
        precision = 0 if num == 0 else num / den
        precision_sum += precision
        i += 1
    return round(precision_sum / len(matrix), 5)


def get_f1_score(precision, recall):
    if precision == 0:
        return 0
    return round(2 * (precision * recall) / (precision + recall), 5)


def get_mean(arr):
    return round(sum(arr) / len(arr), 5)
