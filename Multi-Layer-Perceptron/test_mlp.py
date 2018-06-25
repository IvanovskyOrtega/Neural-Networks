import mlp as mlp
import numpy as np 

'''
Create a MLP network for function aproximation indicating some hyperparameters:
    Architecture: 1,5,5,1 (The first number is the size of the input, the rest are the number of neurons per layer)
    Transfer functions: 2,2,3 (1.logsig, 2.tansig, 3.purelin, corresponding to each layer)
    Learning rate: 0.1
    Epochs: 10000
    Itval: 200 (How often an iteration of validation will be done, for early-stopping algorithm)
    Learning error: 0.0000001 (the maximum error per epoch, also known as threshold)
    Validation error: 0.0000001 (the maximum error in an iteration of validation)
    Increments of v_err: 4 (The maximum amount of consecutive errors in an iteration of validation)
'''
mlp1 = mlp.MLP([1,5,5,1],[2,2,3],0.1,10000,200,0.0000001,0.00000001,4)

# Set the training set indicating the percentage used for the training set.
# The rest will be used for the validation set.
mlp1.set_training_set("patterns.txt","targets.txt",80)

# Train the network
mlp1.train()

# You can save your network if it had a successful training.
mlp1.save_network('nn.pkl')

# You can load your trained network
mlp2 = mlp.MLP.load_network('nn.pkl')

# You can test the network with some values.
res = mlp2.feed_forward_propagate(0.5)
print('The output of the network: '+str(res))

# You can see the performance of the network in a plot
mlp2.show_network_results('faprox')


'''
Create a MLP network for classifying task indicating some hyperparameters:
    Architecture: 1,2,2
    Transfer functions: 1,1
    Learning rate: 0.1
    Epochs: 10000
    Itval: 200 
    Learning error: 0.0000001
    Validation error: 0.0000001
    Increments of v_err: 4 (This value is meaningless if you set the training set to the 100%)
'''
mlp3 = mlp.MLP([2,2,1],[1,1],0.1,10000,200,0.0000001,0.00000001,4)
# Set the training set indicating that the training set will be the 100%
# of the original training set. So, there's no validation set.
mlp3.set_training_set("xor_patterns.txt","xor_targets.txt",100)

# Train the network to solve the XOR classification.
mlp3.train()

# You can save your network if it had a successful training.
mlp3.save_network('nn_xor.pkl')

# You can load your trained network
mlp4 = mlp.MLP.load_network('nn_xor.pkl')

# You can test the network with some values.
res = mlp4.feed_forward_propagate(np.array([0,1]))
print('The output of the network: '+str(res))

# You can see the performance of the network in a plot
mlp4.show_network_results('classif')
