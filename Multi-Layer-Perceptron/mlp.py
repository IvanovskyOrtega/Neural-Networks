from matplotlib import cm
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt
import transferFunctions as tfun

class MLP:
    

    mlp_learning_error = 0.0
    mlp_validation_error = 0.0
    increments = 0


    def __init__(self,arch,tf,alpha,it_max,it_val,learning_error,validation_error,max_inc):
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
        learning_error: float
            The minimum error per epoch in the learning process.
        validation_error: float
            The minimum error in the validation test.
        max_inc: int
            The maximum number of consecutive increments in the validation error.
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
        self.learning_error = learning_error
        self.validation_error = validation_error
        self.max_inc = max_inc


    @staticmethod
    def init_weights(arch):
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
           w = np.random.uniform(-1,1,(arch[i],arch[i-1]))
           W.append(w)
        return W 
    

    @staticmethod
    def init_biases(arch):
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
           b = np.random.uniform(-1,1,(arch[i],1))
           B.append(b)
        return B

    def set_training_set(self,patterns_file,targets_file,t_set_percentage):
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
        config:
            The configuration to generate the training and
            evaluation sets.
        '''
        patterns = np.transpose(np.loadtxt(patterns_file))
        targets = np.transpose(np.loadtxt(targets_file))
        for pattern in patterns:
            if len(pattern.shape) > 1:
                pattern.shape = (pattern.shape[0],1)
        self.patterns = patterns
        self.targets = targets
        self.t_set_percentage = t_set_percentage
        if len(patterns.shape) > 1:
            num_patterns = patterns.shape[1]
        else:
            num_patterns = patterns.shape[0]
        t_set_elements = num_patterns*t_set_percentage//100
        indexes = np.arange(num_patterns)
        t_set = np.random.choice(indexes,t_set_elements,replace=False)
        v_set = np.setdiff1d(indexes,t_set)
        patterns_t = []
        targets_t = []
        patterns_v = []
        targets_v = []
        if len(patterns.shape) > 1:
            for i in t_set:
                patterns_t.append(patterns[:,i])
                targets_t.append(targets[i])
            for i in v_set:
                patterns_v.append(patterns[:,i])
                targets_v.append(targets[i])
        else:
            for i in t_set:
                patterns_t.append(patterns[i])
                targets_t.append(targets[i])
            for i in v_set:
                patterns_v.append(patterns[i])
                targets_v.append(targets[i])
        self.patterns_t = np.array(patterns_t)
        self.targets_t = np.array(targets_t)
        self.patterns_v = np.array(patterns_v)
        self.targets_v = np.array(targets_v)

    @staticmethod
    def transfer_function(type,n):
        '''
        This functions evaluates the value 'n' into its corresponding
        transfer function according to the architecture previously 
        defined.
        Parameters
        ----------
        type: int
            The type of transfer function:
            1-logsig, 2-tansig, 3-purelin
        n: Float, Array
            The result of the operation (W*a + b)
        Returns:
            The value/array evaluated into the function.
        '''
        if type == 1:
            return tfun.logsig(n)
        elif type == 2:
            return tfun.tansig(n)
        elif type == 3:
            return tfun.purelin(n)


    def propagate_patterns(self):
        '''
        This functions propagates forward all the patterns of the training
        set into all the layers of the network.
        '''
        j = 0
        MLP.mlp_learning_error = 0.0
        for pattern in self.patterns_t:
            a = pattern
            self.layer_output[0] = a
            for i in range(0,len(self.W)):
                n = np.dot(self.W[i], a) + self.B[i]
                a = MLP.transfer_function(self.tf[i],n)
                self.layer_output[i+1] = a
            error = (self.targets_t[j]-a)
            MLP.mlp_learning_error += error
            self.backpropagation(error)
            j += 1
        MLP.mlp_learning_error = MLP.mlp_learning_error / self.patterns_t.shape[0]

    def show_network_results(self,type):
        '''
        This functions shows the performance of the network result of 
        the training process.
        Parameters
        ----------
        type: String
            Indicates if the MLP was used for classification or as a
            function aproximator. Posible options:
                - classif : Indicates classificaction.
                - faprox : Indicates function aproximator.
        '''
        j = 0
        MLP.learning_error = 0.0
        outputs = []
        for pattern in self.patterns_t:
            a = pattern
            for i in range(0,len(self.W)):
                n = np.dot(self.W[i], a) + self.B[i]
                a = MLP.transfer_function(self.tf[i],n)
            # print('Pattern: '+str(pattern)+', Output: '+str(a))
            outputs.append(a[0])
        outputs = np.array(outputs)
        if type == 'classif':
            self.plot_classif()
        elif type == 'faprox':
            self.plot_function(outputs)


    def plot_function(self,outputs):
        '''
        This function plots the target output vs the output of the network.
        Parameters
        ----------
        outputs: Array
            The array of results of the network.
        '''
        plt.title("MLP Function Aproximator")
        plt.plot(self.patterns,self.targets,color='black')
        plt.scatter(self.patterns_t,outputs,color='red',edgecolors='black')
        plt.show()


    def plot_classif(self):
        '''
        This function plots the result of the network as a classifier.
        It plots the input patterns as scatter and the fronter boundaries
        as lines.
        '''
        plt.title("MLP Classifier")
        plt.grid(color='gray', linestyle='-', linewidth=0.5)
        x_min, x_max = self.patterns_t[:, 0].min() - 1, self.patterns_t[:, 0].max() + 1
        y_min, y_max = self.patterns_t[:, 1].min() - 1, self.patterns_t[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),np.arange(y_min, y_max, 0.02))
        test_set = np.c_[xx.ravel(), yy.ravel()]
        Z = []
        for element in test_set:
            Z.append(self.feed_forward_propagate(element))
        Z = np.array(Z)
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, cmap=plt.cm.plasma)
        plt.scatter(self.patterns_t[:,0],self.patterns_t[:,1],color='red',edgecolors='black')
        plt.show()

    def validate(self):
        '''
        This functions performs an iteration of validation
        through all the elements of the validation set.
        '''
        if self.t_set_percentage == 100:
            return
        v_error = 0.0
        j = 0
        for pattern in self.patterns_v:
            a = pattern
            for i in range(0,len(self.W)):
                n = np.dot(self.W[i], a) + self.B[i]
                a = MLP.transfer_function(self.tf[i],n)
            error = (self.targets_v[j]-a)
            v_error += error
            j += 1
        v_error = v_error / self.patterns_v.shape[0]
        if v_error > MLP.mlp_validation_error:
            MLP.increments += 1
        else:
            MLP.increments = 0

    def early_stopping(self):
        '''
        This function verifies if the early stopping condition is
        satisfied.
        '''
        if MLP.increments == self.max_inc:
            return True
        else:
            return False


    def is_trained(self):
        '''
        This function evaluates if the neural network is trained
        (probably trained) according to the learning error value.
        If the learning error (learning_error) is less than the
        previosly defined value or it is greater or equal than 0
        then, we can conclude the training process.
        '''
        if np.sum(MLP.mlp_learning_error) < self.learning_error and np.sum(MLP.mlp_learning_error) >= 0:
            return True
        else:
            return False

    @staticmethod
    def get_transfer_function_derivative(type,a):
        '''
        This function gets the derivative of the transfer function 
        of a layer according to its output value (for fast
        performance).
        Parameters
        ----------
        type: int
            The type of transfer function
        a: float, array
            The output value of the layer
        Returns
        -------
        derivative: float, array
            The derivative of the transfer function evaluated in a.
        '''
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
        '''
        This function calculate the sensitivities of each layer used
        in the backpropagation algorithm.
        Parameters
        ----------
        error: float, array
            The difference between the target and the network output.
        '''
        M = self.layers-1
        F = MLP.get_transfer_function_derivative(self.tf[M],self.layer_output[M+1])
        self.S[M] = -2 * F * error
        for m in range(M-1,-1,-1):
            F = MLP.get_transfer_function_derivative(self.tf[m],self.layer_output[m+1])
            W_T = np.transpose(self.W[m+1])
            self.S[m] = np.dot(F,np.dot(W_T,self.S[m+1]))
    

    def adjust_weights_and_biases(self):
        '''
        This function performs the learning process of the network (adjust
        weights and biases for each layer). 
        '''
        M = self.layers
        for m in range(0,M):
            a_t = np.transpose(self.layer_output[m])
            self.W[m] = self.W[m] - (self.alpha * np.dot(self.S[m],a_t))
            self.B[m] = self.B[m] - (self.alpha * self.S[m])


    def backpropagation(self,error):
        '''
        This function corresponds to the learning rule of the network
        aka Backpropgation.
        '''
        self.calculate_sensitivities(error)
        self.adjust_weights_and_biases()

    def train(self):
        '''
        This function performs the training process of the network.
        '''
        for i in range(0,self.it_max):
            print('Iteration: '+str(i))
            self.propagate_patterns()
            if (i % self.it_val) == 0:
                self.validate()
                if self.early_stopping():
                    print('Iteration: '+str(i+1))
                    print('Training finished by early stopping')
                    break
            if self.is_trained():
                print('Iteration: '+str(i+1))
                print('The network had a successful training')
                break
        if i == self.it_max-1:
            print("The network achieved it_max")


    def feed_forward_propagate(self,input_pattern):
        '''
        This function propagates an input into a previously trained MLP
        to see the performance of the network.
        Parameters
        ----------
        input_pattern: Float, Array
            The pattern to test within the network.
        '''
        a = input_pattern
        for i in range(0,len(self.W)):
            n = np.dot(self.W[i], a) + self.B[i]
            a = MLP.transfer_function(self.tf[i],n)
        return a

    
    def save_network(self,filename):
        '''
        This function saves the trained network and writes it to a file
        for further testing or application using Pickler.
        Parameters
        ----------
        filename: String
            The name of the file to save the network. It will overwrite if
            exist, so be careful!
        '''
        with open(filename, 'wb') as output:
            pkl.dump(self, output, pkl.HIGHEST_PROTOCOL)


    @staticmethod
    def load_network(filename):
        '''
        This function loads a trained network from a file using Pickler.
        Parameters
        ----------
        filename: String
            The name of the file in which the network was saved.
        '''
        trained_network = None
        with open(filename, 'rb') as input:
            trained_network = pkl.load(input)
        return trained_network
        

if __name__ == '__main__':
    print('Check test_mlp.py for usage, or read the Documentation.')
