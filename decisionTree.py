import math
import random
import matplotlib.pyplot as plot
import sklearn
import copy


class DecisionTree:
    def __init__(self, instances, labels, num_attr, cat_index_n_values, class_index, criterion):
        self.instances = instances
        self.num_attr = num_attr
        self.class_index = class_index
        self.cat_index_n_values = cat_index_n_values
        self.gini = True if criterion == 'gini' else False
        self.labels = labels
        self.root = self.get_node(instances, 0, 0, [])

    # returns the dic with key: label and value:count(represents no. of times it present in the instances)
    def get_label_map(self, instances):
        label_map = {}
        for inst in instances:
            label = inst[self.class_index]
            if label not in label_map:
                label_map[label] = 1
            else:
                label_map[label] += 1
        return label_map

    # returns the gini value
    def get_gini(self, instances):
        instances_count = len(instances)
        label_map = self.get_label_map(instances)
        prob_square_sum = 0
        for label in label_map:
            prob = label_map[label] / instances_count
            prob_square_sum += prob ** 2
        return 1 - prob_square_sum

    # returns the entropy value
    def get_entropy(self, instances):
        instances_count = len(instances)
        label_map = self.get_label_map(instances)
        entropy = 0
        for label in label_map:
            prob = label_map[label] / instances_count
            entropy += -prob * math.log2(prob)
        return entropy

    def get_node(self, data_instances, level, gain, parent_attr):
        result = None
        criterion_value = None
        child = None
        threshold = None
        parent_instance_count = len(data_instances)
        parent_entropy = self.get_entropy(data_instances) if self.gini is False else None
        instance_length = len(data_instances[0])
        parent_attr = copy.copy(parent_attr)

        # return {'label': self.get_max_label(data_instances)}

        # iterating attr_indexes
        for index in range(instance_length):
            if index == self.class_index or index in parent_attr:
                continue
            curr_threshold = None
            temp_child = {}
            # attr is numerical
            if index in self.num_attr:
                curr_threshold = get_numerical_attribute_threshold(index, data_instances)
                lte = []
                gt = []
                for inst in data_instances:
                    value = float(inst[index])
                    if value <= curr_threshold:
                        lte.append(inst)
                    else:
                        gt.append(inst)
                temp_child['lte'] = lte
                temp_child['gt'] = gt
            # attr is categorical
            else:
                values = self.cat_index_n_values[index]
                # initialising map with key: categorical value and empty array
                for val in values:
                    temp_child[val] = []
                for inst in data_instances:
                    value = inst[index]
                    temp_child[value].append(inst)

            # calculating entropy for node
            node_entropy = None
            # iterate with child
            for key in temp_child:
                child_inst = temp_child[key]
                child_inst_length = len(child_inst)
                if child_inst_length == 0:
                    node_entropy = None
                    break
                else:
                    entropy = self.get_entropy(child_inst) if self.gini is False else self.get_gini(child_inst)
                    if node_entropy is None:
                        node_entropy = 0
                    node_entropy += (child_inst_length / parent_instance_count) * entropy

            # updating variables child and criterion value based on node_entropy
            if node_entropy is not None:
                if self.gini:
                    if criterion_value is None or node_entropy < criterion_value:
                        criterion_value = node_entropy
                        result = index
                        child = temp_child
                        threshold = curr_threshold
                else:
                    gain = parent_entropy - node_entropy
                    if gain > 0:
                        if criterion_value is None or gain > criterion_value:
                            criterion_value = gain
                            result = index
                            child = temp_child
                            threshold = curr_threshold

        # no further divided needed
        if child is None:
            return {'label': self.get_max_label(data_instances)}

        for key in child:
            instances = child[key]
            parent_attr.append(result)
            child[key] = self.get_node(instances, level + 1, criterion_value, parent_attr)

        return {
            'index': result,
            'child': child,
            'threshold': threshold,
            'level': level,
            'instances_count': len(data_instances),
            'gain': criterion_value
        }

    # get label with maximum value in the instances
    def get_max_label(self, instances):
        label_map = self.get_label_map(instances)
        result_label = None
        for label in label_map:
            if result_label is None or label_map[label] > label_map[result_label]:
                result_label = label
        return result_label

    # returns the label for an instance for the constructed tree
    def get_predicted_label(self, instance):
        node = self.root
        while 'label' not in node:
            index = node['index']
            val = instance[index]
            if index in self.num_attr:
                val = float(val)
                child_key = 'lte' if val <= node['threshold'] else 'gt'
                node = node['child'][child_key]
            else:
                node = node['child'][val]
        return node['label']

    # returns confusion matrix
    def get_confusion_matrix(self, testing_instances):
        labels = self.labels
        # i*j --> i is inst label and j is predicted label
        matrix = [[0 for i in labels] for i in labels]
        label_index_map = {labels[i]: i for i in range(len(labels))}

        for inst in testing_instances:
            predicted_label = self.get_predicted_label(inst)
            inst_label = inst[self.class_index]
            i = label_index_map[inst_label]
            j = label_index_map[predicted_label]
            matrix[i][j] += 1
        return matrix


# returns the threshold value
def get_numerical_attribute_threshold(attr_index, instances):
    total_sum = 0
    for instance in instances:
        total_sum += float(instance[attr_index])
    return total_sum / len(instances)


# returns dic with key: attr_index and value: arr<cat_values>
def get_cat_values_map(instances, class_index, num_attr):
    cat_attr_map = {}
    for instance in instances:
        for index in range(0, len(instance)):
            if index == class_index or index in num_attr:
                continue
            attr_value = instance[index]
            if index not in cat_attr_map:
                cat_attr_map[index] = [attr_value]
            else:
                attr_values = cat_attr_map[index]
                if attr_value not in attr_values:
                    attr_values.append(attr_value)
    return cat_attr_map


# returns folds
def create_folds(instances, class_index, k):
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


# returns class labels as array
def get_labels(instances, class_index):
    arr = []
    for inst in instances:
        label = inst[class_index]
        if label in arr:
            continue
        else:
            arr.append(label)
    return arr


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


def get_standard_deviation(arr, mean):
    num = 0
    for data in arr:
        num += (data - mean) ** 2
    return round((num / len(arr)) ** (1 / 2), 5)


# entry point for getting the performance graphs
def execute(dataset_name, instances, class_index, num_attr, criterion):
    print("Dataset name:", dataset_name)
    n_folds = 10
    cat_index_n_values = get_cat_values_map(instances, class_index, num_attr)
    folds = create_folds(instances, class_index, n_folds)
    labels = get_labels(instances, class_index)

    acc = []
    precision = []
    recall = []
    f1_score = []
    for testing_index in range(n_folds):
        testing_set = folds[testing_index]
        training_set = []
        for index in range(len(folds)):
            if index != testing_index:
                training_set.extend(folds[index])

        dTree = DecisionTree(training_set, labels, num_attr, cat_index_n_values, class_index, criterion)
        matrix = dTree.get_confusion_matrix(testing_set)
        precision_value = get_precision(matrix)
        recall_value = get_recall(matrix)
        acc.append(get_accuracy(matrix))
        recall.append(recall_value)
        precision.append(precision_value)
        f1_score.append(get_f1_score(precision_value, recall_value))
    print(get_mean(acc))
    print(get_mean(f1_score))
