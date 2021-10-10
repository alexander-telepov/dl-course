# Don't erase the template code, except "Your code here" comments.

import torch
import math                      # Pi

""" Task 1 """

def get_rho():
    pi = math.pi
    theta = torch.linspace(-pi, pi, 1000, dtype=torch.float64)
    assert theta.shape == (1000,)

    rho = (1 + 0.9 * torch.cos(8 * theta)) * (1 + 0.1 * torch.cos(24 * theta)) * \
          (0.9 + 0.05 * torch.cos(200 * theta)) * (1 + torch.sin(theta))
    assert torch.is_same_size(rho, theta)
    
    x = rho * torch.cos(theta)
    y = rho * torch.sin(theta)

    return x, y

""" Task 2 """

def game_of_life_update_torch(alive_map):
    """
    PyTorch version of `game_of_life_update_reference()`.
    
    alive_map:
        `torch.tensor`, ndim == 2, dtype == `torch.int64`
        The game map containing 0s (dead) an 1s (alive).
    """
    kernel = torch.ones(1, 1, 3, 3, dtype=torch.float32)
    with torch.no_grad():
        neigh_map = torch.conv2d(alive_map[None, None, ...].float(), kernel, padding=1).type(torch.int64).squeeze()

    dead = alive_map == 0
    alive = alive_map == 1
    alive_map[dead] = (neigh_map[dead] == 3).type(torch.int64)
    alive_map[alive] = ((neigh_map[alive] == 4) | (neigh_map[alive] == 3)).type(torch.int64)


""" Task 3 """

# This is a reference layout for encapsulating your neural network. You can add arguments and
# methods if you need to. For example, you may want to add a method `do_gradient_step()` that
# executes one step of an optimization algorithm (SGD / Adadelta / Adam / ...); or you can
# add an extra argument for which you'll find a good value during experiments and set it as
# default to preserve interface (e.g. `def __init__(self, num_hidden_neurons=100):`).
class NeuralNet:
    def __init__(self):
        self.W1 = ((torch.rand(28*28, 100, dtype=torch.float32) - 0.5) * 2 / math.sqrt(28*28)).requires_grad_()
        self.b1 = torch.ones(100, dtype=torch.float32, requires_grad=True)
        self.W2 = ((torch.rand(100, 10, dtype=torch.float32) - 0.5) * 2 / math.sqrt(100)).requires_grad_()
        self.b2 = torch.ones(10, dtype=torch.float32, requires_grad=True)
        self.vars = [self.W1, self.W2, self.b1, self.b2]
        self.deltas = list(map(torch.zeros_like, self.vars))

    def predict(self, images):
        """
        images:
            `torch.tensor`, shape == `batch_size x height x width`, dtype == `torch.float32`
            A minibatch of images -- the input to the neural net.
        
        return:
        prediction:
            `torch.tensor`, shape == `batch_size x 10`, dtype == `torch.float32`
            The scores of each input image to belong to each of the dataset classes.
            Namely, `prediction[i, j]` is the score of `i`-th minibatch sample to
            belong to `j`-th class.
            These scores can be 0..1 probabilities, but for better numerical stability
            they can also be raw class scores after the last (usually linear) layer,
            i.e. BEFORE softmax.
        """
        return torch.relu(images.view(-1, 28*28) @ self.W1 + self.b1) @ self.W2 + self.b2

    def do_gradient_step(self, lr=0.1, momentum=0.9):
        for i, var in enumerate(self.vars):
            self.deltas[i] = momentum * self.deltas[i] - lr * var.grad.data
            var.data += self.deltas[i]

        for var in self.vars:
            var.grad.data = torch.zeros_like(var.grad.data)

def accuracy(model, images, labels):
    """
    Use `NeuralNet.predict` here.
    
    model:
        `NeuralNet`
    images:
        `torch.tensor`, shape == `N x height x width`, dtype == `torch.float32`
    labels:
        `torch.tensor`, shape == `N`, dtype == `torch.int64`
        Class indices for each sample in `images`.
    
    return:
    value:
        `float`
        The fraction of samples from `images` correctly classified by `model`.
        `0 <= value <= 1`.
    """
    with torch.no_grad():
        predictions = model.predict(images)

    return (torch.argmax(predictions, dim=1) == labels).sum() / len(labels)

def train_on_notmnist(model, X_train, y_train, X_val, y_val):
    """
    Update `model`'s weights so that its accuracy on `X_val` is >=82%.
    `X_val`, `y_val` are provided for convenience and aren't required to be used.
    
    model:
        `NeuralNet`
    X_train:
        `torch.tensor`, shape == `N x height x width`, dtype == `torch.float32`
    y_train:
        `torch.tensor`, shape == `N`, dtype == `torch.int64`
        Class indices for each sample in `X_train`.
    X_val, y_val:
        Same as above, possibly with a different length.
    """
    for i in range(len(X_train) // 100):
        X, y_true = X_train[i*100:(i+1)*100], y_train[i*100:(i+1)*100]
        y_pred = model.predict(X)
        loss = -torch.log_softmax(y_pred, dim=1)[torch.arange(100), y_true].mean()
        loss.backward()
        model.do_gradient_step()

