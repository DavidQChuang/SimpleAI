#pragma once

namespace nn {
	class PerceptronTrainer : public SupervisedNetworkTrainer {
	private:
		double deltaW(double target, double result, double input) {
			return learningRate * (target - result) * input;
		}

	public:
		void trainInplace(NeuralNetwork network,
			double* inputs, size_t inLength, double* outputs, size_t outLength) override {
			if (network.getLayers().size() != 0)
				throw invalid_argument("Can only train networks with 1 layer.");
			
			double *buffer, *networkOutputs;
			executeNetwork(network, buffer, networkOutputs, inputs, inLength);


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