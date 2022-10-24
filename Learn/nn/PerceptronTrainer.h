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

		void trainInplace(NeuralNetwork network,
			double* inputs, size_t inLength, double* outputs, size_t outLength) {
			if (network.getLayers().size() > 2)
				throw invalid_argument("Can only train networks with 1 inout layer or 1 input and 1 output layer. ");
			if (outLength != network.expectedOutputs()) {
				throw invalid_argument("Invalid number of outputs.");
			}

			double* buffer, *outPtr;

			NeuronLayer layer = network.getLayers()[0];
			Neuron* neurons = layer.getNeurons();
			size_t neuronCount = layer.size();

			printf("\nInputs: [");
			for (int i = 0;;) {
				printf(to_string(inputs[i]).c_str());

				if (++i < inLength) {
					printf(", ");
				}
				else break;
			}
			printf("]");

			printf("\n\nExpected outputs: [");
			for (int i = 0;;) {
				printf(to_string(outputs[i]).c_str());

				if (++i < outLength) {
					printf(", ");
				}
				else break;
			}
			printf("]");

			int e = 0;
			double mse = 0;
			for (e = 0; e < epochTarget; e++) {
				//printf("\n\n### Epoch #%s", to_string(e).c_str());

				executeNetwork(network, buffer, outPtr, inputs, inLength);

				mse = cost(outLength, outPtr, outputs);
				/*printf("\nNN Outputs: [");
				for (int i = 0;;) {
					printf(to_string(outPtr[i]).c_str());

					if (++i < outLength) {
						printf(", ");
					}
					else break;
				}
				printf("]");
				printf("\n\nMean Standard Error: [ %s ]\n", to_string(mse).c_str());*/

				if (mse < errorTarget) {
					break;
				}

				for (int i = 0; i < neuronCount; i++) {
					Neuron neuron = neurons[i];

					vector<double> weightsIn = neuron.weightsIn();
					double error = outputs[i] - outPtr[i]; // target - result

					weightsIn[0] += deltaW(error, weightsIn[0]);
				}
			}

			if (e == epochTarget) {
				printf("\nReached epoch limit on epoch %s with an MSE of %s\n", to_string(e).c_str(), to_string(mse).c_str());
			}
			else {
				printf("\nSucceeded on epoch %s with an MSE of %s\n", to_string(e).c_str(), to_string(mse).c_str());
			}
		}
	};
}