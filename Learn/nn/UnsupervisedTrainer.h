#pragma once

#include <cmath>
#include "../NeuralNetwork.h"

namespace nn {
	class UnsupervisedTrainer {
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

		virtual void trainOnEpoch(NeuralNetwork& network, double* inputs, double* buffer, double* outPtr) = 0;

		virtual bool checkTrainingInputs(NeuralNetwork& network, double* inputs, size_t inLength) {
			if (network.expectedInputs() != inLength)
				throw invalid_argument("Input of network and size of input buffer don't match.");

			return true;
		}

	public:
		UnsupervisedTrainer(double learnRate = 0.1, double error = 0.002, int epochs = 1000) {
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

		void train(NeuralNetwork& network, int trainingSets, double** inputs, size_t inLength) {

			int bufferSize = network.expectedBufferSize();

			unique_ptr<double[]> bufferPtr(new double[bufferSize]);
			double* buffer = bufferPtr.get();

			for (int i = 0; i < trainingSets; i++) {
				printf("\n\n### Training set #%d\n", i);

				trainOnSet(network, buffer, inputs[i], inLength);
			}
		}

		void train(NeuralNetwork& network, double* inputs, size_t inLength) {

			int bufferSize = network.expectedBufferSize();

			unique_ptr<double[]> bufferPtr(new double[bufferSize]);
			double* buffer = bufferPtr.get();

			trainOnSet(network, buffer, inputs, inLength);
		}

	private:
		void trainOnSet(NeuralNetwork& network, double* buffer, double* inputs, size_t inLength) {
			if (!checkTrainingInputs(network, inputs, inLength)) {
				return;
			}

			printf("\n%-10s | [ ", "Inputs");
			for (int i = 0;;) {
				printf("%.3f", inputs[i]);

				if (++i < inLength) {
					printf(", ");
				}
				else break;
			}
			printf(" ]");

			int e = 0;
			double mse = 0;

			int bufferSize = network.expectedBufferSize();
			double* outPtr = nullptr;
			memcpy(buffer, inputs, inLength * sizeof(double));

			for (e = 0; e < epochTarget; e++) {
				outPtr = network.executeToIOArray(buffer, inLength, bufferSize);

				trainOnEpoch(network, inputs, buffer, outPtr);
			}

			int outputCount = network.expectedOutputs();
			printf("\n%-10s | [ ", "NNOutputs");
			for (int i = 0;;) {
				printf("%.6f", outPtr[i]);

				if (++i < outputCount) {
					printf(", ");
				}
				else break;
			}
			printf(" ]");
			printf("\n");

			if (e == epochTarget) {
				printf("%-10s | %-30s | Epoch %-3d", "Result", "Reached epoch limit", e);
			}
			else {
				printf("%-10s | %-30s | Epoch %-3d", "Result", "Reached minimum rate limit", e);
			}
			printf("\n");
		}
	};
}