import numpy as np
import perceptron as p
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Train perceptron or predict using trained perceptron')
    parser.add_argument('-t', help='Train[t] or predict[p]', required=True)
    parser.add_argument('-d', help='Dataset path', required=True)
    parser.add_argument('-m', help='Path to load or save model. Usage: path/to/model', required=True)
    args = parser.parse_args()

    return args.t, args.d, args.m

def load(path):

    data = np.load(path)
    X = data['X']
    Y = data['Y']

    return X, Y

def save_model(nn, model_path):
    np.savez_compressed(model_path, w=nn.W)

def load_model(nn, model_path):
    W = np.load(model_path)['w']
    nn.W = W

def train(nn, X, Y):
    
    convergence = 0
    num_inputs =  X.shape[0]

    # Train until convergence is achieved
    num_training = 0
    while convergence != num_inputs:
        for x, y in zip(X, Y):
            nn.predict(x)
            correct = nn.update(x, y)
            if correct:
                convergence +=1 

        if convergence < num_inputs:
            convergence = 0
            num_training += 1

    print(f'This perceptron was trained for {num_training} epochs')

def predict(nn, X, Y):
    
    correct = 0
    # Predict labels
    for x, y in zip(X, Y):
        y_pred = nn.predict(x)
        if y_pred == y:
            correct += 1
    
    # Print accuracy
    acc = correct/X.shape[0]
    print(f'The accuracy of this perceptron is: {acc*100} %')

def main(**kwargs):
    
    # Parse arguments
    type, data_path, model_path = parse_args()
    
    # Load data
    X, Y = load(data_path)
    
    # Get input and output shape
    in_size = X.shape[1]
    out_size = 1

    # Initialize perceptron
    nn = p.Perceptron(in_size, out_size)

    # Train or predict
    if type == 't':
        train(nn, X, Y)
        save_model(nn, model_path)
    else:
        load_model(nn, model_path)
        predict(nn, X, Y)

if __name__ == '__main__':
    main()
