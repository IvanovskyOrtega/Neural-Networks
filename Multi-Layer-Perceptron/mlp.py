import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt

class MLP:
    

    mlp_e_ap = 0.0
    mlp_e_val = 0.0
    increments = 0.0


    def __init__(self,arch,tf,alpha,it_max,it_val,e_ap,e_val):
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
        alpha: float
            The learning rate for the MLP.
        it_max: int
            Maximum number of epochs of the MLP.
        it_val: int
            How often an iteration of validation will be done.
        e_ap: float
            The minimum error per epoch in the learning process.
        e_val: float
            The minimum error in the validation test.
        '''
        self.arch = arch
        self.tf = tf
        self.R = arch[0]
        self.layers = len(arch)-1
        self.W = MLP.init_weights(arch)
        self.B = MLP.init_biases(arch)
        self.layer_output = [None] * (self.layers + 1)
        self.S = [None] * self.layers
        self.alpha = alpha
        self.it_max = it_max
        self.it_val = it_val
        self.e_ap = e_ap
        self.e_val = e_val


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
           print(w)
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
        The file must be a text file (*.txt).

        Parameters
        ----------
        patterns_file: String
            The filename or path of the patterns "*.txt" file
        targets_file: String
            The filename or path of the tagets "*.txt" file
        '''
        self.patterns = np.transpose(np.loadtxt(patterns_file))
        self.targets = np.transpose(np.loadtxt(targets_file))

    @classmethod
    def transfer_function(cls,type,x):
        if type == 1:
            return 1 / (1 + np.exp(-x))
        elif type == 2:
            return ( np.exp(x) - np.exp(-x) ) / ( np.exp(x) + np.exp(-x) )
        elif type == 3:
            return x


    def propagate_patterns(self):
        j = 0
        MLP.mlp_e_ap = 0.0
        for pattern in self.patterns:
            a = pattern
            self.layer_output[0] = a
            for i in range(0,len(self.W)):
                n = np.dot(self.W[i], a) + self.B[i]
                a = MLP.transfer_function(self.tf[i],n)
                self.layer_output[i+1] = a
            error = (self.targets[j]-a)
            MLP.mlp_e_ap += error
            self.backpropagation(error)
            j += 1
        print('Error b: '+str(MLP.mlp_e_ap))
        MLP.mlp_e_ap = MLP.mlp_e_ap / self.patterns.shape[0]

    def show_network_results(self):
        j = 0
        MLP.e_ap = 0.0
        outputs = []
        for pattern in self.patterns:
            a = pattern
            for i in range(0,len(self.W)):
                n = np.dot(self.W[i], a) + self.B[i]
                a = MLP.transfer_function(self.tf[i],n)
            print('Pattern: '+str(pattern)+', Output: '+str(a))
            outputs.append(a[0])
        outputs = np.array(outputs)
        plt.plot(self.patterns,self.targets)
        plt.xlabel("Desired function")
        plt.plot(self.patterns,outputs)
        plt.show()


    def validate(self):
        print("I'm validating")
    

    def early_stopping(self):
        print("Early Stopping")


    def is_trained(self):
        print(np.sum(MLP.mlp_e_ap))
        print(self.e_ap)
        if np.sum(MLP.mlp_e_ap) < self.e_ap and np.sum(MLP.mlp_e_ap) >= 0:
            return True
        else:
            return False

    @classmethod
    def get_transfer_function_derivative(cls,type,a):
        if np.isscalar(a):
            rows = 1
        else:
            rows = a.shape[0]
        if type == 1:
            ones = np.ones((rows,1))
            diag = np.diagflat(ones-a)
            derivative = np.dot(diag,np.diagflat(a))
            return derivative
        elif type == 2:
            ones = np.ones((rows,1))
            sqr = np.square(a)
            sub = ones-sqr
            derivative = np.diagflat(sub)
            return derivative
        elif type == 3:
            derivative = np.diagflat(np.ones((rows,1)))
            return derivative


    def calculate_sensitivities(self,error):
        M = self.layers-1
        F = MLP.get_transfer_function_derivative(self.tf[M],self.layer_output[M+1])
        self.S[M] = -2 * F * error
        for m in range(M-1,-1,-1):
            F = MLP.get_transfer_function_derivative(self.tf[m],self.layer_output[m+1])
            W_T = np.transpose(self.W[m+1])
            self.S[m] = np.dot(F,np.dot(W_T,self.S[m+1]))
    

    def adjust_weights_and_biases(self):
        M = self.layers
        for m in range(0,M):
            a_t = np.transpose(self.layer_output[m])
            self.W[m] = self.W[m] - (self.alpha * np.dot(self.S[m],a_t))
            self.B[m] = self.B[m] - (self.alpha * self.S[m])


    def backpropagation(self,error):
        self.calculate_sensitivities(error)
        self.adjust_weights_and_biases()

    def train(self):
        for i in range(0,self.it_max):
            print('Iteration: '+str(i+1))
            self.propagate_patterns()
            if(i % self.it_val):
                self.validate() #
                self.early_stopping() #
            if self.is_trained():
                print('The network had a successful training')
                break
        if i == self.it_max:
            print("The network achieved it_max")
        self.show_network_results() # 

mlp = MLP([1,6,4,4,1],[2,2,1,3],0.2,5000,20,0.00000001,0.00000001)
mlp.set_training_set("patterns.txt","targets.txt")
mlp.train()