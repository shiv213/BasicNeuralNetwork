import numpy as np

class NeuralNetwork():
	def __init__(self):
        # seeding for random number generation
		np.random.seed(1)
		
		# converting weights to a 3 by 1 matric with values from -1 to 1 and a mean of 0
		self.synaptic_weights = 2 * np.random.random((3, 1)) - 1
	
	def sigmoid(self, x):
		# apply the Sigmoid function
		return 1 / (1 + np.exp(-x))

	def sigmoid_derivative(self, x):
		# compute derivative to the Sigmoid function
		return x * (1 - x)
	
	def train(self, training_inputs, training_outputs, training_iterations):
		# training the model to make accurate predictions while adjusting weights continually
		for i in range(training_iterations):
			# siphon the training data via the neuron
			output = self.think(training_inputs)

			# compute error rate for back propagation
			error = training_outputs - output

			# performing weight adjustments
			adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))

			self.synaptic_weights += adjustments


	def think(self, inputs):
		# passing the inputs via the neuron to get output
		# converting values to floats
		
		inputs = inputs.astype(float)
		output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
		return output

if __name__ == "__main__":
	# instantiate the neuron class
	myNetwork = NeuralNetwork()

	print("beginning randomly generated weights: ")
	print(myNetwork.synaptic_weights)
	
	# training data consisting of 4 samples (3 input, 1 output)
	training_inputs = np.array([[0,0,1],
								[1,1,1],
								[1,0,1],
								[0,1,1]])

	training_outputs = np.array([[0,1,1,0]]).T

	# training taking place
	myNetwork.train(training_inputs, training_outputs, 15000)

	print("weights after training: ")
	print(myNetwork.synaptic_weights)

	print("considering new case: 1 0 0")
	print("new output:")
	print(myNetwork.think(np.array([1, 0, 0])))
	