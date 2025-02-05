import csv
import time
from collections import defaultdict
from statistics import stdev
from utils import *

with open('abalone.csv', newline='') as csvfile:
    data = list(csv.reader(csvfile))

[i.pop(0) for i in data]
for i in range(len(data)):
    data[i] = [float(j) for j in data[i]]

with open('abalonetest.csv', newline='') as csvfile:
    testdata = list(csv.reader(csvfile))

[i.pop(0) for i in testdata]
for i in range(len(testdata)):
    testdata[i] = [float(j) for j in testdata[i]]


# From aima code
class DataSet:
    """
    A data set for a machine learning problem. It has the following fields:

    d.examples   A list of examples. Each one is a list of attribute values.
    d.attrs      A list of integers to index into an example, so example[attr]
                 gives a value. Normally the same as range(len(d.examples[0])).
    d.attr_names Optional list of mnemonic names for corresponding attrs.
    d.target     The attribute that a learning algorithm will try to predict.
                 By default the final attribute.
    d.inputs     The list of attrs without the target.
    d.values     A list of lists: each sublist is the set of possible
                 values for the corresponding attribute. If initially None,
                 it is computed from the known examples by self.set_problem.
                 If not None, an erroneous value raises ValueError.
    d.distance   A function from a pair of examples to a non-negative number.
                 Should be symmetric, etc. Defaults to mean_boolean_error
                 since that can handle any field types.
    d.name       Name of the data set (for output display only).
    d.source     URL or other source where the data came from.
    d.exclude    A list of attribute indexes to exclude from d.inputs. Elements
                 of this list can either be integers (attrs) or attr_names.

    Normally, you call the constructor and you're done; then you just
    access fields like d.examples and d.target and d.inputs.
    """

    def __init__(self, examples=None, attrs=None, attr_names=None, target=-1, inputs=None,
                 values=None, distance=mean_boolean_error, name='', source='', exclude=()):
        """
        Accepts any of DataSet's fields. Examples can also be a
        string or file from which to parse examples using parse_csv.
        Optional parameter: exclude, as documented in .set_problem().

        <DataSet(): 1 examples, 3 attributes>
        """
        self.name = name
        self.source = source
        self.values = values
        self.distance = distance
        self.got_values_flag = bool(values)

        # initialize .examples from string or list or data directory
        if isinstance(examples, str):
            self.examples = parse_csv(examples)
        elif examples is None:
            self.examples = parse_csv(open_data(name + '.csv').read())
        else:
            self.examples = examples

        # attrs are the indices of examples, unless otherwise stated.
        if self.examples is not None and attrs is None:
            attrs = list(range(len(self.examples[0])))

        self.attrs = attrs

        # initialize .attr_names from string, list, or by default
        if isinstance(attr_names, str):
            self.attr_names = attr_names.split()
        else:
            self.attr_names = attr_names or attrs
        self.set_problem(target, inputs=inputs, exclude=exclude)

    def set_problem(self, target, inputs=None, exclude=()):
        """
        Set (or change) the target and/or inputs.
        This way, one DataSet can be used multiple ways. inputs, if specified,
        is a list of attributes, or specify exclude as a list of attributes
        to not use in inputs. Attributes can be -n .. n, or an attr_name.
        Also computes the list of possible values, if that wasn't done yet.
        """
        self.target = self.attr_num(target)
        exclude = list(map(self.attr_num, exclude))
        if inputs:
            self.inputs = remove_all(self.target, inputs)
        else:
            self.inputs = [a for a in self.attrs if a != self.target and a not in exclude]
        if not self.values:
            self.update_values()
        self.check_me()

    def check_me(self):
        """Check that my fields make sense."""
        assert len(self.attr_names) == len(self.attrs)
        assert self.target in self.attrs
        assert self.target not in self.inputs
        assert set(self.inputs).issubset(set(self.attrs))
        if self.got_values_flag:
            # only check if values are provided while initializing DataSet
            list(map(self.check_example, self.examples))

    def add_example(self, example):
        """Add an example to the list of examples, checking it first."""
        self.check_example(example)
        self.examples.append(example)

    def check_example(self, example):
        """Raise ValueError if example has any invalid values."""
        if self.values:
            for a in self.attrs:
                if example[a] not in self.values[a]:
                    raise ValueError('Bad value {} for attribute {} in {}'
                                     .format(example[a], self.attr_names[a], example))

    def attr_num(self, attr):
        """Returns the number used for attr, which can be a name, or -n .. n-1."""
        if isinstance(attr, str):
            return self.attr_names.index(attr)
        elif attr < 0:
            return len(self.attrs) + attr
        else:
            return attr

    def update_values(self):
        self.values = list(map(unique, zip(*self.examples)))

    def sanitize(self, example):
        """Return a copy of example, with non-input attributes replaced by None."""
        return [attr_i if i in self.inputs else None for i, attr_i in enumerate(example)]

    def classes_to_numbers(self, classes=None):
        """Converts class names to numbers."""
        if not classes:
            # if classes were not given, extract them from values
            classes = sorted(self.values[self.target])
        for item in self.examples:
            item[self.target] = classes.index(item[self.target])

    def remove_examples(self, value=''):
        """Remove examples that contain given value."""
        self.examples = [x for x in self.examples if value not in x]
        self.update_values()

    def split_values_by_classes(self):
        """Split values into buckets according to their class."""
        buckets = defaultdict(lambda: [])
        target_names = self.values[self.target]

        for v in self.examples:
            item = [a for a in v if a not in target_names]  # remove target from item
            buckets[v[self.target]].append(item)  # add item to bucket of its class

        return buckets

    def find_means_and_deviations(self):
        """
        Finds the means and standard deviations of self.dataset.
        means     : a dictionary for each class/target. Holds a list of the means
                    of the features for the class.
        deviations: a dictionary for each class/target. Holds a list of the sample
                    standard deviations of the features for the class.
        """
        target_names = self.values[self.target]
        feature_numbers = len(self.inputs)

        item_buckets = self.split_values_by_classes()

        means = defaultdict(lambda: [0] * feature_numbers)
        deviations = defaultdict(lambda: [0] * feature_numbers)

        for t in target_names:
            # find all the item feature values for item in class t
            features = [[] for _ in range(feature_numbers)]
            for item in item_buckets[t]:
                for i in range(feature_numbers):
                    features[i].append(item[i])

            # calculate means and deviations fo the class
            for i in range(feature_numbers):
                means[t][i] = mean(features[i])
                deviations[t][i] = stdev(features[i])

        return means, deviations

    def __repr__(self):
        return '<DataSet({}): {:d} examples, {:d} attributes>'.format(self.name, len(self.examples), len(self.attrs))


def NearestNeighborLearner(dataset, k=1):
    """k-NearestNeighbor: the k nearest neighbors vote."""

    def predict(example):
        """Find the k closest items, and have them vote for the best."""
        best = heapq.nsmallest(k, ((dataset.distance(e, example), e) for e in dataset.examples))
        return mode(e[dataset.target] for (d, e) in best)

    return predict


def parse_csv(input, delim=','):
    r"""
    Input is a string consisting of lines, each line has comma-delimited
    fields. Convert this into a list of lists. Blank lines are skipped.
    Fields that look like numbers are converted to numbers.
    The delim defaults to ',' but '\t' and None are also reasonable values.

    [[1, 2, 3], [0, 2, 'na']]
    """
    lines = [line for line in input.splitlines() if line.strip()]
    return [list(map(num_or_str, line.split(delim))) for line in lines]


class CountingProbDist:
    """
    A probability distribution formed by observing and counting examples.
    If p is an instance of this class and o is an observed value, then
    there are 3 main operations:
    p.add(o) increments the count for observation o by 1.
    p.sample() returns a random element from the distribution.
    p[o] returns the probability for o (as in a regular ProbDist).
    """

    def __init__(self, observations=None, default=0):
        """
        Create a distribution, and optionally add in some observations.
        By default this is an unsmoothed distribution, but saying default=1,
        for example, gives you add-one smoothing.
        """
        if observations is None:
            observations = []
        self.dictionary = {}
        self.n_obs = 0
        self.default = default
        self.sampler = None

        for o in observations:
            self.add(o)

    def add(self, o):
        """Add an observation o to the distribution."""
        self.smooth_for(o)
        self.dictionary[o] += 1
        self.n_obs += 1
        self.sampler = None

    def smooth_for(self, o):
        """
        Include o among the possible observations, whether or not
        it's been observed yet.
        """
        if o not in self.dictionary:
            self.dictionary[o] = self.default
            self.n_obs += self.default
            self.sampler = None

    def __getitem__(self, item):
        """Return an estimate of the probability of item."""
        self.smooth_for(item)
        return self.dictionary[item] / self.n_obs

    # (top() and sample() are not used in this module, but elsewhere.)

    def top(self, n):
        """Return (count, obs) tuples for the n most frequent observations."""
        return heapq.nlargest(n, [(v, k) for (k, v) in self.dictionary.items()])

    def sample(self):
        """Return a random sample from the distribution."""
        if self.sampler is None:
            self.sampler = weighted_sampler(list(self.dictionary.keys()), list(self.dictionary.values()))
        return self.sampler()


def NaiveBayesDiscrete(dataset):
    """
    Just count how many times each value of each input attribute
    occurs, conditional on the target value. Count the different
    target values too.
    """

    target_vals = dataset.values[dataset.target]
    target_dist = CountingProbDist(target_vals)
    attr_dists = {(gv, attr): CountingProbDist(dataset.values[attr]) for gv in target_vals for attr in dataset.inputs}
    for example in dataset.examples:
        target_val = example[dataset.target]
        target_dist.add(target_val)
        for attr in dataset.inputs:
            attr_dists[target_val, attr].add(example[attr])

    def predict(example):
        """
        Predict the target value for example. Consider each possible value,
        and pick the most likely by looking at each attribute independently.
        """

        def class_probability(target_val):
            return (target_dist[target_val] * product(attr_dists[target_val, attr][example[attr]]
                                                      for attr in dataset.inputs))

        return max(target_vals, key=class_probability)

    return predict


class NNUnit:
    """
    Single Unit of Multiple Layer Neural Network
    inputs: Incoming connections
    weights: Weights to incoming connections
    """

    def __init__(self, activation=sigmoid, weights=None, inputs=None):
        self.weights = weights or []
        self.inputs = inputs or []
        self.value = None
        self.activation = activation


def init_examples(examples, idx_i, idx_t, o_units):
    inputs, targets = {}, {}

    for i, e in enumerate(examples):
        # input values of e
        inputs[i] = [e[i] for i in idx_i]

        if o_units > 1:
            # one-hot representation of e's target
            t = [0 for i in range(o_units)]
            t[e[idx_t]] = 1
            targets[i] = t
        else:
            # target value of e
            targets[i] = [e[idx_t]]

    return inputs, targets


def BackPropagationLearner(dataset, net, learning_rate, epochs, activation=sigmoid):
    """
    [Figure 18.23]
    The back-propagation algorithm for multilayer networks.
    """
    # initialise weights
    for layer in net:
        for node in layer:
            node.weights = random_weights(min_value=-0.5, max_value=0.5, num_weights=len(node.weights))

    examples = dataset.examples
    # As of now dataset.target gives an int instead of list,
    # Changing dataset class will have effect on all the learners.
    # Will be taken care of later.
    o_nodes = net[-1]
    i_nodes = net[0]
    o_units = len(o_nodes)
    idx_t = dataset.target
    idx_i = dataset.inputs
    n_layers = len(net)

    inputs, targets = init_examples(examples, idx_i, idx_t, o_units)

    for epoch in range(epochs):
        # iterate over each example
        for e in range(len(examples)):
            i_val = inputs[e]
            t_val = targets[e]

            # activate input layer
            for v, n in zip(i_val, i_nodes):
                n.value = v

            # forward pass
            for layer in net[1:]:
                for node in layer:
                    inc = [n.value for n in node.inputs]
                    in_val = dot_product(inc, node.weights)
                    node.value = node.activation(in_val)

            # initialize delta
            delta = [[] for _ in range(n_layers)]

            # compute outer layer delta

            # error for the MSE cost function
            err = [t_val[i] - o_nodes[i].value for i in range(o_units)]

            # calculate delta at output
            if node.activation == sigmoid:
                delta[-1] = [sigmoid_derivative(o_nodes[i].value) * err[i] for i in range(o_units)]
            elif node.activation == relu:
                delta[-1] = [relu_derivative(o_nodes[i].value) * err[i] for i in range(o_units)]
            elif node.activation == tanh:
                delta[-1] = [tanh_derivative(o_nodes[i].value) * err[i] for i in range(o_units)]
            elif node.activation == elu:
                delta[-1] = [elu_derivative(o_nodes[i].value) * err[i] for i in range(o_units)]
            elif node.activation == leaky_relu:
                delta[-1] = [leaky_relu_derivative(o_nodes[i].value) * err[i] for i in range(o_units)]
            else:
                return ValueError("Activation function unknown.")

            # backward pass
            h_layers = n_layers - 2
            for i in range(h_layers, 0, -1):
                layer = net[i]
                h_units = len(layer)
                nx_layer = net[i + 1]

                # weights from each ith layer node to each i + 1th layer node
                w = [[node.weights[k] for node in nx_layer] for k in range(h_units)]

                if activation == sigmoid:
                    delta[i] = [sigmoid_derivative(layer[j].value) * dot_product(w[j], delta[i + 1])
                                for j in range(h_units)]
                elif activation == relu:
                    delta[i] = [relu_derivative(layer[j].value) * dot_product(w[j], delta[i + 1])
                                for j in range(h_units)]
                elif activation == tanh:
                    delta[i] = [tanh_derivative(layer[j].value) * dot_product(w[j], delta[i + 1])
                                for j in range(h_units)]
                elif activation == elu:
                    delta[i] = [elu_derivative(layer[j].value) * dot_product(w[j], delta[i + 1])
                                for j in range(h_units)]
                elif activation == leaky_relu:
                    delta[i] = [leaky_relu_derivative(layer[j].value) * dot_product(w[j], delta[i + 1])
                                for j in range(h_units)]
                else:
                    return ValueError("Activation function unknown.")

            # update weights
            for i in range(1, n_layers):
                layer = net[i]
                inc = [node.value for node in net[i - 1]]
                units = len(layer)
                for j in range(units):
                    layer[j].weights = vector_add(layer[j].weights,
                                                  scalar_vector_product(learning_rate * delta[i][j], inc))

    return net


def network(input_units, hidden_layer_sizes, output_units, activation=sigmoid):
    """
    Create Directed Acyclic Network of given number layers.
    hidden_layers_sizes : List number of neuron units in each hidden layer
    excluding input and output layers
    """
    layers_sizes = [input_units] + hidden_layer_sizes + [output_units]

    net = [[NNUnit(activation) for _ in range(size)] for size in layers_sizes]
    n_layers = len(net)

    # make connection
    for i in range(1, n_layers):
        for n in net[i]:
            for k in net[i - 1]:
                n.inputs.append(k)
                n.weights.append(0)
    return net


def find_max_node(nodes):
    return nodes.index(max(nodes, key=lambda node: node.value))


def PerceptronLearner(dataset, learning_rate=0.01, epochs=100):
    """Logistic Regression, NO hidden layer"""
    i_units = len(dataset.inputs)
    o_units = len(dataset.values[dataset.target])
    hidden_layer_sizes = []
    raw_net = network(i_units, hidden_layer_sizes, o_units)
    learned_net = BackPropagationLearner(dataset, raw_net, learning_rate, epochs)

    def predict(example):
        o_nodes = learned_net[1]

        # forward pass
        for node in o_nodes:
            in_val = dot_product(example, node.weights)
            node.value = node.activation(in_val)

        # hypothesis
        return find_max_node(o_nodes)

    return predict


#######################################################################################################################
# My Implementation


def useKNN():
    start = time.time()
    kNN = NearestNeighborLearner(abalone, k=25)
    abalone.classes_to_numbers()

    print("Test  0: Prediction Actual")
    for i in range(len(testdata)):
        print(f"Test {i + 1:>2}: {kNN(testdata[i]):>5}, {data[i][-1]:>7}")
    end = time.time()
    print(f"Total time for kNN: {(end - start):.2f}s.")  # 26.44s


def useNBD():
    start = time.time()
    nBD = NaiveBayesDiscrete(abalone)
    abalone.classes_to_numbers()

    print("Test  0: Prediction Actual")
    for i in range(len(testdata)):
        print(f"Test {i + 1:>2}: {nBD(testdata[i]):>5}, {data[i][-1]:>7}")
    end = time.time()
    print(f"Total time for NBD: {(end - start):.2f}s.") # 0.23s


def usePerceptron():
    start = time.time()
    abalone.classes_to_numbers()
    perceptron = PerceptronLearner(abalone)

    print("Test  0: Prediction Actual")
    for i in range(len(testdata)):
        print(f"Test {i + 1:>2}: {perceptron(testdata[i]):>5}, {data[i][-1]:>7}")
    end = time.time()
    print(f"Total time for NBD: {(end - start):.2f}s.") # 67.41s


abalone = DataSet(name="abalone", examples=data)


def main():
    global abalone

    useKNN()
    #useNBD()
    #usePerceptron()

    return


if __name__ == "__main__":
    main()
