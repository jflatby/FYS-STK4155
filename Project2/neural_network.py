import numpy as np

def batch_splitting(inputs, targets, number_of_batches=15, randomize=True):
    if randomize:
        mask = np.random.shuffle(np.arange(len(inputs)))
        inputs = inputs[mask]
        targets = targets[mask]

    indeces = np.linspace(0, len(inputs), number_of_batches+1, dtype=int)

    for i in range(number_of_batches):
        batch_inputs = inputs[indeces[i]:indeces[i+1]]
        batch_targets = targets[indeces[i]:indeces[i+1]]

        yield batch_inputs, batch_targets
