from numpy import exp, array, random, dot

class NeuralNetwork():
    def __init__(self):
        random.seed(1)
        
        #Neural Network structure
        self.synaptic_weights1 = 2* random.random((3, 6)) - 1
        self.synaptic_weights2 = 2* random.random((6, 4)) - 1
        self.synaptic_weights3 = 2* random.random((4, 3)) - 1
        self.synaptic_weights4 = 2* random.random((3, 1)) - 1
    
    #activator
    def __sigmoid(self, x):
        return 1 / (1 + exp(-x))
    
    #check with gradient descent and re-adjust using slope
    def __sigmoid_derivative(self, x):
        return x * (1 - x)
    
    #train the neural network
    def train(self, training_set_inputs, training_set_outputs, number_of_iterations):
        #iterate 10,000 times
        for i in range(number_of_iterations):
            #pass training set through neural network
            l1 = self.__sigmoid(dot(training_set_inputs, self.synaptic_weights1))
            l2 = self.__sigmoid(dot(l1, self.synaptic_weights2))
            l3 = self.__sigmoid(dot(l2, self.synaptic_weights3))
            output = self.__sigmoid(dot(l3, self.synaptic_weights4))
            
            #error of set
            e_l1 = (training_set_outputs - output) * self.__sigmoid_derivative(output)
            
            #error of each layer
            e_l4 = dot(self.synaptic_weights4, e_l1.T) * (self.__sigmoid_derivative(l3).T)
            e_l3 = dot(self.synaptic_weights3, e_l4) * (self.__sigmoid_derivative(l2).T)
            e_l2 = dot(self.synaptic_weights2, e_l3) * (self.__sigmoid_derivative(l1).T)
            
            #adjust using error and gradient of slope
            adjustment4 = dot(l3.T, e_l1)
            adjustment3 = dot(l2.T, e_l4.T)
            adjustment2 = dot(l1.T, e_l3.T)
            adjustment1 = dot(training_set_inputs.T, e_l2.T)
            
            self.synaptic_weights1 += adjustment1
            self.synaptic_weights2 += adjustment2
            self.synaptic_weights3 += adjustment3
            self.synaptic_weights4 += adjustment4
 
    #perdict using network given input value        
    def predict(self, input):
        l1 = self.__sigmoid(dot(input, self.synaptic_weights1))
        l2 = self.__sigmoid(dot(l1, self.synaptic_weights2))
        l3 = self.__sigmoid(dot(l2, self.synaptic_weights3))
        output = self.__sigmoid(dot(l3, self.synaptic_weights4))
        
        return output
        
if __name__ == "__main__":
    
    neural_network = NeuralNetwork()
    
    print("Random starting synaptic weights (layer 1): ")
    print(neural_network.synaptic_weights1)
    print("Random starting synaptic weights (layer 2): ")
    print(neural_network.synaptic_weights2)
    print("Random starting synaptic weights (layer 3): ")
    print(neural_network.synaptic_weights3)
    print("Random starting syntapic weights (layer 4): ")
    print(neural_network.synaptic_weights4)
    
    training_set_inputs = array([[0,0,1], [1,1,1], [1,0,1], [0,1,1]])
    training_set_outputs = array([[0,1,1,0]]).T
    
    neural_network.train(training_set_inputs, training_set_outputs, 10000)
    
    print("New synaptic weights after training (layer 1): ")
    print(neural_network.synaptic_weights1)
    print("New synaptic weights after training (layer 2): ")
    print(neural_network.synaptic_weights2)
    print("New synaptic weights after training (layer 3): ")
    print(neural_network.synaptic_weights3)
    print("new synpatic weights after training (layer 4): ")
    print(neural_network.synaptic_weights4)
    
    print("predict [1, 1, 0]:")
    print(neural_network.predict([1, 1, 0]))
    
