#pragma once

namespace nn {
	class PerceptronTrainer : public SupervisedNetworkTrainer {
	private:
		double deltaW(double error, double input) {
			return learningRate * error * input;
		}

	public:
		void trainInplace(NeuralNetwork network,
			double* inputs, size_t inLength, double* outputs, size_t outLength) override {
			if (network.getLayers().size() != 0)
				throw invalid_argument("Can only train networks with 1 layer.");
			
			double *buffer, *outPtr;

			auto layer = network.getLayers()[0];
			auto neurons = layer.getNeurons();

			if (outLength != neurons.size())
				throw invalid_argument("Can only train networks with as many outputs as neurons.");

			for (int e = 0; e < epochTarget; e++) {
				executeNetwork(network, buffer, outPtr, inputs, inLength);

				double mse = cost(neurons.size(), outPtr, outputs);
				printf("Mean Standard Error: [ %s ]", to_string(mse).c_str());

				if (mse < errorTarget) break;

				for (int i = 0; i < neurons.size(); i++) {
					auto neuron = neurons[i];

					double prevW = neuron.weightsOut()[0];
					double error = outputs[i] - outPtr[i]; // target - result

					neuron.weightsOut()[0] = prevW + deltaW(error, prevW);
				}
			}
		}
	};

	class SupervisedNetworkTrainer {
	protected:
		int				epochTarget;
		double			errorTarget;
		double			learningRate;

		double cost(int n, double* nnEstimate, double* actual) {
			double sum = 0;

			for (int i = 0; i < n; i++) {
				sum += pow(nnEstimate[i] - actual[i], 2);
			}
			
			return 1 / n * sum;
		}

		// 
		//double deltaW(double target, double result, double input, double adalineDeriv) {
		//	// rate * (target - result) * input
		//	switch (trainingType) {
		//	case Perceptron:
		//		return learningRate * (target - result) * input;
		//	case Adaline:
		//		return learningRate * (target - result) * input * adalineDeriv;
		//	}
		//}

	protected:
		void executeNetwork(NeuralNetwork network, double*& buffer, double*& networkOutputs, double* inputs, size_t inLength) {
			size_t bufferSize = network.expectedBufferSize();
			buffer = new double[bufferSize];

			memcpy(buffer, inputs, inLength);
			networkOutputs = network.executeToIOArray(buffer, inLength, bufferSize);
		}

		//bool 

	public:
		SupervisedNetworkTrainer(double learnRate = 0.1, double error = 0.002, int epochs = 1000) {
			learningRate = learnRate;
			errorTarget = error;
			epochTarget = epochs;
		}

		void setLearningRate(double rate) {
			learningRate = rate;
		}

		void setErrorTarget(double error) {
			errorTarget = error;
		}

		void setEpochTarget(int epochs) {
			epochTarget = epochs;
		}

		virtual void trainInplace(NeuralNetwork network, 
			double* inputs, size_t inLength, double* outputs, size_t outLength) = 0;

		NeuralNetwork trainCopy(NeuralNetwork network, double* inputs, size_t inLength, double* outputs, size_t outLength) {
			NeuralNetwork newNet = NeuralNetwork(network);
			trainInplace(newNet, inputs, inLength, outputs, outLength);
			return newNet;
		}
	};
};