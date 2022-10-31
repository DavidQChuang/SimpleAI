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

		virtual void train(NeuralNetwork& network, double* inputs, double* expOutputs, double* buffer, double* outPtr) = 0;
		virtual bool checkTrainingInputs(NeuralNetwork& network,
			double* inputs, size_t inLength, double* expOutputs, size_t outLength) {
			if (network.expectedInputs() != inLength)
				throw invalid_argument("Input of network and size of input buffer don't match.");
			if (network.expectedOutputs() != outLength)
				throw invalid_argument("Output of network and size of output buffer don't match.");

			return true;
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

		void trainInplace(NeuralNetwork& network,
			double* inputs, size_t inLength, double* expOutputs, size_t outLength) {
			if (!checkTrainingInputs(network, inputs, inLength, expOutputs, outLength)) {
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

			printf("\n%-10s | [ ", "ExpOutputs");
			for (int i = 0;;) {
				printf("%.6f", expOutputs[i]);

				if (++i < outLength) {
					printf(", ");
				}
				else break;
			}
			printf(" ]");

			int bufferSize = network.expectedBufferSize();

			unique_ptr<double[]> bufferPtr( new double[bufferSize] );
			double* buffer = bufferPtr.get(),
			      * outPtr = nullptr;

			int e = 0;
			double mse = 0;

			memcpy(buffer, inputs, inLength * sizeof(double));

			for (e = 0; e < epochTarget; e++) {
				outPtr = network.executeToIOArray(buffer, inLength, bufferSize);

				mse = cost(outLength, outPtr, expOutputs);
				if (mse < errorTarget) {
					break;
				}

				train(network, inputs, expOutputs, buffer, outPtr);
			}

			printf("\n%-10s | [ ", "NNOutputs");
			for (int i = 0;;) {
				printf("%.6f", outPtr[i]);

				if (++i < outLength) {
					printf(", ");
				}
				else break;
			}
			printf(" ]");
			printf("\n%-10s | [ %.6e ]\n", "MSError", mse);

			if (e == epochTarget) {
				printf("%-10s | %-30s | Epoch %-3d", "Result", "Failed - Reached epoch limit", e);
			}
			else {
				printf("%-10s | %-30s | Epoch %-3d", "Result", "Succeeded - reached MSE limit", e);
			}
			printf("\n");
		}

		NeuralNetwork trainCopy(NeuralNetwork network, double* inputs, size_t inLength, double* outputs, size_t outLength) {
			trainInplace(network, inputs, inLength, outputs, outLength);
			return network;
		}
	};
};