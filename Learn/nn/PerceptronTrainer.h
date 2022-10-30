#pragma once

#include "SupervisedNetworkTrainer.h"

namespace nn {

	class PerceptronTrainer : public SupervisedNetworkTrainer {
	private:
		double deltaW(double error, double input) {
			return learningRate * error * input;
		}

	public:
		PerceptronTrainer(double learnRate = 0.1, double error = 0.002, int epochs = 1000)
			: SupervisedNetworkTrainer(learnRate, error, epochs) {

		}

		void trainInplace(NeuralNetwork& network,
			double* inputs, size_t inLength, double* expOutputs, size_t outLength) {
			if (network.getLayers().size() > 2)
				throw invalid_argument("Perceptron requires 1 inout layer or 1 in + 1 out layer. ");

			if (outLength != network.expectedOutputs())
				throw invalid_argument("Invalid number of outputs.");

			if (outLength != 1)
				throw invalid_argument("Perceptron requires 1 output.");

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

			vector<NeuronLayer>& layers = network.getLayers();
			NeuronLayer& layer = layers[0];
			vector<double>& weightsIn = layer.weightsIn();

			int neurons = layer.size();

			double *buffer = nullptr, *outPtr = nullptr;

			int e = 0;
			double mse = 0;
			for (e = 0; e < epochTarget; e++) {
				//printf("\n\n### Epoch #%s", to_string(e).c_str());

				executeNetwork(network, buffer, outPtr, inputs, inLength);

				mse = cost(outLength, outPtr, expOutputs);
				if (mse < errorTarget) {
					break;
				}

				double error = expOutputs[0] - outPtr[0]; // target - result, positive if result was lower, negative if result was higher

				int in = 0;
				for (int n = 0; n < neurons; n++) {
					for (int i = 0; i < layer.neuronInputs(); i++) {
						weightsIn[n] += deltaW(error, inputs[in++]);
					}
				}
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
			printf("\n%-10s | [ %.6f ]\n", "MSError", mse);

			delete[] buffer;

			if (e == epochTarget) {
				printf("%-10s | %-30s | Epoch %-3d", "Result", "Failed - Reached epoch limit", e);
			}
			else {
				printf("%-10s | %-30s | Epoch %-3d", "Result", "Succeeded - reached MSE limit", e);
			}
			printf("\n");
		}
	};
}