#pragma once

#include "SupervisedNetworkTrainer.h"

namespace nn {
	class AdalineTrainer : public SupervisedNetworkTrainer {
	private:
		NeuronLayer* layer;
		vector<double>* weightsIn;
		int neurons;

	protected:
		bool checkTrainingInputs(NeuralNetwork& network,
			double* inputs, size_t inLength, double* expOutputs, size_t outLength) override {
			bool success = SupervisedNetworkTrainer::checkTrainingInputs(network, inputs, inLength, expOutputs, outLength);

			if (network.getLayers().size() > 2)
				throw invalid_argument("Adaline requires 1 inout layer or 1 in + 1 out layer. ");

			if (outLength != 1)
				throw invalid_argument("Adaline requires 1 output.");

			layer = &network.getLayers()[0];
			weightsIn = &layer->weightsIn();
			neurons = layer->size();

			return success;
		}

		void trainOnEpoch(NeuralNetwork& network, double* inputs, double* expOutputs, double* buffer, double* outPtr) {
			double error = expOutputs[0] - outPtr[0]; // target - result, positive if result was lower, negative if result was higher
			double sum = 0;

			int inputCount = layer->inputsPerNeuron();
			int outputCount = layer->outputsPerNeuron();

			int inOffset = network.getLayers().size() == 1 ? 0 : network.expectedInputs();
			double* inPtr = buffer + inOffset;

			for (int n = 0; n < neurons; n++) {
				for (int o = 0; o < outputCount; o++) {
					sum += inPtr[n * outputCount + o];
				}
			}

			int in = 0;
			for (int n = 0; n < neurons; n++) {
				for (int i = 0; i < inputCount; i++) {
					weightsIn[0][n] += learningRate * error * inputs[in] * layer->derivActivationFunc(sum);

					in++;
				}
			}
		}

	public:
		AdalineTrainer(double learnRate = 0.1, double error = 0.002, int epochs = 1000)
			: SupervisedNetworkTrainer(learnRate, error, epochs) { }
	};
}