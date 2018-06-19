import numpy as np

class MLP:
    

    def __init__(self,arch,tf):
        '''
        Constructor for the Multi Layer Perceptron

        Parameters
        -----------
        arch: List
            A list which contains the size of the input (R) and the number of neurons per layer.
            Each element in the list is considered as a layer.
            Defines the architecture (topology).
        tf: List
            A list which contains the transfer functions used for each layer.
        '''
        self.arch = arch
        self.tf = tf
        self.R = arch[0]
        self.W = MLP.init_weights(arch)
        self.B = MLP.init_biases(arch)


    @classmethod
    def init_weights(cls,arch):
        '''
        This function initialize all the weight matrices of the MLP.

        Parameters
        ----------
        arch: List
            The architecture of the neural network.

        Returns
        ----------
        W: List
            A list which contains the weight matrices for each layer.
            The values for each entry of the matrix are random numbers 
            between 0 and 1.
        '''
        W = []
        for i in range(1,len(arch)):
           w = np.random.random((arch[i],arch[i-1]))
           W.append(w)
        return W 
    

    @classmethod
    def init_biases(cls,arch):
        '''
        This function initialize all the bias vectors of the MLP.

        Parameters
        ----------
        arch: List
            The architecture of the neural network.

        Returns
        ----------
        B: List
            A list which contains the bias vectors for each layer.
            The values for each element in the vector are random numbers 
            between 0 and 1.
        '''
        B = []
        for i in range(1,len(arch)):
           b = np.random.random((arch[i],1))
           B.append(b)
        return B

    def set_training_set(self,patterns_file,targets_file):
        '''
        This function loads the training set from two separated files, one
        for patterns, and one for targets.
        The file must be a text file (.txt).

        Parameters
        ----------
        patterns_file: String
            The filename or path of the patterns ".txt" file
        targets_file: String
            The filename or path of the tagets ".txt" file
        '''
        self.patterns = np.loadtxt(filename)
        self.targets = np.loadtxt(filename)

mlp = MLP([4,2,3],[1,2])