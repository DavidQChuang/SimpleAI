#pragma once

#include "SupervisedTrainer.h"

namespace nn {
	class PerceptronTrainer : public SupervisedTrainer {
	private:
		NeuronLayer* layer;
		vector<double>* weightsIn;
		int neurons;

	protected:
		bool checkTrainingInputs(NeuralNetwork& network,
			double* inputs, size_t inLength, double* expOutputs, size_t outLength) override {
			bool success = SupervisedTrainer::checkTrainingInputs(network, inputs, inLength, expOutputs, outLength);

			if (network.getLayers().size() > 2)
				throw invalid_argument("Perceptron trainer requires 1 inout layer or 1 in + 1 out layer. ");

			if (outLength != 1)
				throw invalid_argument("Perceptron requires 1 output.");

			layer = &network.getLayers()[0];
			weightsIn = &layer->weightsIn();
			neurons = layer->size();

			return success;
		}

		void trainOnEpoch(NeuralNetwork& network, double* inputs, double* expOutputs, double* buffer, double* outPtr) {
			double error = expOutputs[0] - outPtr[0]; // target - result, positive if result was lower, negative if result was higher

			int in = 0;
			for (int n = 0; n < neurons; n++) {
				for (int i = 0; i < layer->inputsPerNeuron(); i++) {
					(*weightsIn)[n] += learningRate * error * inputs[in++];
				}
			}
		}

	public:
		PerceptronTrainer(double learnRate = 0.1, double error = 0.002, int epochs = 1000)
			: SupervisedTrainer(learnRate, error, epochs) { }
	};
}