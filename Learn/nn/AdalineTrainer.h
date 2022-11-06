#pragma once

#include "SupervisedTrainer.h"

namespace nn {
	class AdalineTrainer : public SupervisedTrainer {
	private:
		NeuronLayer* layer;
		vector<double>* weightsIn;
		int neurons;

	protected:
		void checkTrainingInputs(NeuralNetwork& network,
			double* inputs, size_t inLength, double* expOutputs, size_t outLength) override {
			SupervisedTrainer::checkTrainingInputs(network, inputs, inLength, expOutputs, outLength);

			if (network.getLayers().size() > 2)
				throw invalid_argument("Adaline requires 1 inout layer or 1 in + 1 out layer. ");

			if (outLength != 1)
				throw invalid_argument("Adaline requires 1 output.");

			layer = &network.getLayers()[0];
			weightsIn = &layer->weightsIn();
			neurons = layer->size();
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
					(*weightsIn)[n] += learningRate * error * inputs[in] * layer->derivActivationFunc(sum);

					in++;
				}
			}
		}

	public:
		AdalineTrainer(double learnRate = 0.1, double error = 0.002, int epochs = 1000)
			: SupervisedTrainer(learnRate, error, epochs) { }
	};
}