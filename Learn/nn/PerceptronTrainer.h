#pragma once

#include "SupervisedTrainer.h"

namespace nn {
	template<typename... LayerArgs>
	class PerceptronTrainer : public SupervisedTrainer<LayerArgs...> {
	private:
		NeuralNetwork::Layer* layer;
		vector<double>* weightsInPtr;
		int neurons;

	protected:
		void initTrainingSet(FFNeuralNetwork<LayerArgs...>& network,
			double* inputs, size_t inLength,
			double* expOutputs, size_t outLength)
			override {
			SupervisedTrainer<LayerArgs...>::initTrainingSet(network, inputs, inLength, expOutputs, outLength);

			if (network.depth() > 2)
				throw invalid_argument("Perceptron trainer requires 1 inout layer or 1 in + 1 out layer. ");

			if (outLength != 1)
				throw invalid_argument("Perceptron requires 1 output.");

			layer = &network.getLayer(0);
			weightsInPtr = &layer->weightsIn();
			neurons = layer->size();

			//inputOffset = network.getLayers().size() == 1 ? 0 : network.expectedInputs();
		}

		void trainOnSet(FFNeuralNetwork<LayerArgs...>& network,
			double* inputs, double* expOutputs,
			double* buffer, double* outPtr)
			override {
			double error = expOutputs[0] - outPtr[0]; // target - result, positive if result was lower, negative if result was higher

			int inputCount = layer->inputsPerNeuron();

			vector<double>& weightsIn = *weightsInPtr;

			int in = 0;
			for (int n = 0; n < neurons; n++) {
				for (int i = 0; i < layer->inputsPerNeuron(); i++) {
					int w = n * inputCount + i;

					weightsIn[w] += this->learningRate * error * inputs[in++];
				}
				if (!layer->independentInputs()) {
					in -= inputCount;
				}
			}
		}

	public:
		PerceptronTrainer(double learnRate = 0.1, double error = 0.002, int epochs = 1000)
			: SupervisedTrainer<LayerArgs...>(learnRate, error, epochs) { }
	};
}