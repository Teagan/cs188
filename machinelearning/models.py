import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        # Implement the run(self, x) method. This should compute 
        # the dot product of the stored weight vector and the given 
        # input, returning an nn.DotProduct object.

        # nn.DotProduct(features, weights)

        weights = self.get_weights()
        return nn.DotProduct(x, weights)

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        # Implement get_prediction(self, x), which should return 1 if 
        # the dot product is non-negative or −1 otherwise. You should use 
        # nn.as_scalar to convert a scalar Node into a Python floating-point 
        # number.

        if nn.as_scalar(self.run(x)) < 0:
            return -1
        return 1
        

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        # Write the train(self) method. This should repeatedly loop over 
        # the data set and make updates on examples that are misclassified. 
        # Use the update method of the nn.Parameter class to update the 
        # weights. When an entire pass over the data set is completed without 
        # making any mistakes, 100% training accuracy has been achieved, and 
        # training can terminate.
        # 
        # In this project, the only way to change the value of a parameter is 
        # by calling parameter.update(direction, multiplier), which will perform 
        # the update to the weights: weights = weights + direction * multiplier 
        # The direction argument is a Node with the same shape as the parameter, 
        # and the multiplier argument is a Python scalar.

        # update(self, direction, multiplier)

        # train until no misclassifications in one full sweep
        done = False

        while not done:
            updated = False
            for i in range(len(dataset.x)):
                for x, y in dataset.iterate_once(1):
                    pred = self.get_prediction(x)
                    if pred != nn.as_scalar(y):
                        # print(i, "th loop:\n==================")
                        # print("pred: [", pred, "],   y : [", nn.as_scalar(y), "]")
                        # print("old weights: ", self.get_weights().data)
                        self.get_weights().update(x, nn.as_scalar(y))
                        updated = True
                        # print("updated to: ", self.get_weights().data)
                        # print("==================")
                        break
            if updated == False:
                done = True


class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        # batch_size = 1 
        your_hidden_layer_size = 200
        dimension = 1
        self.w1 = nn.Parameter(dimension, your_hidden_layer_size)
        self.b1 = nn.Parameter(1, your_hidden_layer_size)
        self.w2 = nn.Parameter(your_hidden_layer_size, 1)
        self.b2 = nn.Parameter(1, 1)

        self.learning_rate = 0.01

    def get_w1(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w1

    def get_w2(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w2

    def get_b1(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.b1

    def get_b2(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.b2

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"

        # relu( x * w_1 + b_1 ) * w_2 + b_2

        x_times_W_1 = nn.Linear(x, self.get_w1())
        x_times_W_1_plus_b_1 = nn.AddBias(x_times_W_1, self.get_b1())
        relu_result = nn.ReLU(x_times_W_1_plus_b_1)
        relu_times_W_2 = nn.Linear(relu_result, self.get_w2())
        result = nn.AddBias(relu_times_W_2, self.get_b2())

        return result


    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        y_star = self.run(x)
        return nn.SquareLoss(y_star, y)


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        done = False
        batch_size = 20
        w1 = self.get_w1()
        w2 = self.get_w2()
        b1 = self.get_b1()
        b2 = self.get_b2()
        alpha = -1 * self.learning_rate

        while not done:
            updated = False
            for i in range(len(dataset.x)):
                for x, y in dataset.iterate_once(batch_size):
                    loss = self.get_loss(x, y)
                    if nn.as_scalar(loss) > 0.02:
                        w1grad, w2grad, b1grad, b2grad = nn.gradients(loss, [w1, w2, b1, b2])
                        w1.update(w1grad, alpha)
                        w2.update(w2grad, alpha)
                        b1.update(b1grad, alpha)
                        b2.update(b2grad, alpha)

                        # print("w1:  [", w1.data, "]")
                        # print("w2:  [", w2.data, "]")
                        # print("b1:  [", b1.data, "]")
                        # print("b2:  [", b2.data, "]")
                        updated = True
                        break
            if updated == False:
                done = True



class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        your_hidden_layer_size = 1000
        dimension = 784
        self.w1 = nn.Parameter(dimension, your_hidden_layer_size)
        self.b1 = nn.Parameter(1, your_hidden_layer_size)
        self.w2 = nn.Parameter(your_hidden_layer_size, 10)
        self.b2 = nn.Parameter(1, 10)
        # self.w3 = nn.Parameter(your_hidden_layer_size, 10)
        # self.b3 = nn.Parameter(1, 10)

        self.learning_rate = 0.5

    def get_w1(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w1

    def get_w2(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w2

    # def get_w3(self):
    #     """
    #     Return a Parameter instance with the current weights of the perceptron.
    #     """
    #     return self.w3

    def get_b1(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.b1

    def get_b2(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.b2

    # def get_b3(self):
    #     """
    #     Return a Parameter instance with the current weights of the perceptron.
    #     """
    #     return self.b3

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"

        # relu( x * w_1 + b_1 ) * w_2 + b_2 

        x_times_W_1 = nn.Linear(x, self.get_w1())
        x_times_W_1_plus_b_1 = nn.AddBias(x_times_W_1, self.get_b1())
        relu_result = nn.ReLU(x_times_W_1_plus_b_1)
        relu_times_W_2 = nn.Linear(relu_result, self.get_w2())
        result = nn.AddBias(relu_times_W_2, self.get_b2())

        # temp1 = nn.ReLU(result)
        # temp2 = nn.Linear(temp1, self.get_w3())
        # temp3 = nn.AddBias(temp2, self.get_b3())

        return result

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        y_star = self.run(x)
        return nn.SoftmaxLoss(y_star, y)


    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        done = False
        batch_size = 200
        w1 = self.get_w1()
        w2 = self.get_w2()
        # w3 = self.get_w3()
        b1 = self.get_b1()
        b2 = self.get_b2()
        # b3 = self.get_b3()
        alpha = -1 * self.learning_rate

        while True:
            for x, y in dataset.iterate_once(batch_size):
                if dataset.get_validation_accuracy() > 0.975: # QUICK, QUIT!! IT'S DONE!!
                    return
                loss = self.get_loss(x, y)
                w1grad, b1grad, w2grad, b2grad = nn.gradients(loss, [w1, b1, w2, b2])
                w1.update(w1grad, alpha)
                w2.update(w2grad, alpha)
                b1.update(b1grad, alpha)
                b2.update(b2grad, alpha)

                # print("w1:  [", w1.data, "]")
                # print("w2:  [", w2.data, "]")
                # print("b1:  [", b1.data, "]")
                # print("b2:  [", b2.data, "]")
                # print("current accuracy :  ", dataset.get_validation_accuracy())


class DeepQModel(object):
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.
    """
    def __init__(self, state_dim, action_dim):
        self.num_actions = action_dim
        self.state_size = state_dim

        # Remember to set self.learning_rate, self.numTrainingGames,
        # self.parameters, and self.batch_size!
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.5
        self.numTrainingGames = 800
        self.batch_size = 1000

        your_hidden_layer_size = 1000
        self.w1 = nn.Parameter(state_dim, your_hidden_layer_size)
        self.b1 = nn.Parameter(1, your_hidden_layer_size)
        self.w2 = nn.Parameter(your_hidden_layer_size, action_dim)
        self.b2 = nn.Parameter(1, action_dim)


    def get_loss(self, states, Q_target):
        """
        Returns the Squared Loss between Q values currently predicted 
        by the network, and Q_target.
        Inputs:
            states: a node with shape (batch_size x state_dim)
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            loss node between Q predictions and Q_target
        """
        "*** YOUR CODE HERE ***"
        Q_pred = self.run(states)
        return nn.SquareLoss(Q_pred, Q_target)


    def run(self, states):
        """
        Runs the DQN for a batch of states.
        The DQN takes the state and returns the Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]
        Inputs:
            states: a node with shape (batch_size x state_dim)
        Output:
            result: a node with shape (batch_size x num_actions) containing Q-value
                scores for each of the actions
        """
        "*** YOUR CODE HERE ***"


    def gradient_update(self, states, Q_target):
        """
        Update your parameters by one gradient step with the .update(...) function.
        Inputs:
            states: a node with shape (batch_size x state_dim)
            Q_target: a (batch_size x num_actions) numpy array, or None
        Output:
            None
        """
        "*** YOUR CODE HERE ***"
        w1.update(w1grad, alpha)
