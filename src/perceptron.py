import numpy as np

class Perceptron:
    
    def __init__(self, num_inputs, num_outputs):
        
        # Initialize weights
        self.W = np.random.uniform((num_inputs, num_outputs))
        
        # Initialize prediction
        self.y_pred = np.zeros(num_outputs)
        
    def predict(self, x):
        # Weight input dot product
        y_pred = np.dot(self.W.T, x)
        
        # Neuron activation
        self.y_pred = self.activation(y_pred)

        return self.y_pred

    def activation(self, net_input):
        # Apply step function
        if net_input < 0:
            net_input = 0
        else:
            net_input = 1
        return net_input

    def update(self, x, y_truth):
        if self.y_pred < y_truth:
            self.W += x
            return False
        elif self.y_pred > y_truth:
            self.W -= x
            return False
        else:
            return True

