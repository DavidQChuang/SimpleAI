#pragma once
#include <cmath>
#include "../NeuralNetwork.h"

namespace nn {
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
			
			return sum / n;
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

			if (bufferSize == 0) throw invalid_argument("Network had size 0");
			buffer = new double[bufferSize];

			memcpy(buffer, inputs, inLength * sizeof(double));
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
			NeuralNetwork newNet = NeuralNetwork::copy(network);
			trainInplace(newNet, inputs, inLength, outputs, outLength);
			return newNet;
		}
	};
};