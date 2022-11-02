#pragma once

#include "SupervisedNetworkTrainer.h"

namespace nn {
	class BackpropagationTrainer : public SupervisedNetworkTrainer {
	protected:
		bool checkTrainingInputs(NeuralNetwork& network,
			double* inputs, size_t inLength, double* expOutputs, size_t outLength) override {
			bool success = SupervisedNetworkTrainer::checkTrainingInputs(network, inputs, inLength, expOutputs, outLength);

			if (network.getLayers().size() < 2)
				throw invalid_argument("Backpropagation requires at least 2 layers. ");

			return success;
		}

		void trainOnEpoch(NeuralNetwork& network, double* inputs, double* expOutputs, double* buffer, double* outPtr) {
			vector<double> layerDelta;
			vector<NeuronLayer>& layers = network.getLayers();

			int out = 0;
			NeuronLayer& outputLayer = layers[layers.size() - 1];
			for (int n = 0; n < outputLayer.size(); n++) {
				double sumNN = 0;
				double sumExp = 0;

				for (int o = 0; o < outputLayer.outputsPerNeuron(); o++) {
					sumNN += outPtr[out];
					sumExp += expOutputs[out];
					out++;
				}

				layerDelta.push_back(sumNN - sumExp);
			}

			double* inPtr = outPtr;

			for (int l = layers.size() - 1; l >= 0; l--) {
				NeuronLayer& layer = layers[l];
				vector<double>& weightsIn = layer.weightsIn();

				inPtr -= layer.expectedInputs();

				int inputCount = layer.inputsPerNeuron();
				int in = 0;

				layerDelta.reserve(inputCount);
				for (int i = layerDelta.size(); i < inputCount; i++) {
					layerDelta.push_back(0);
				}

				for (int n = 0; n < layer.size(); n++) {
					int inSum = 0;

					// sum inputs
					for (int i = 0; i < inputCount; i++) {
						inSum += inPtr[in];
						in++;
					}
					in -= inputCount;

					double delta = layerDelta[n] * layer.derivActivationFunc(inSum);

					// adjust weights and set deltas for next layer
					for (int i = 0; i < inputCount; i++) {
						int w = n * inputCount + i;

						layerDelta[i] += delta * weightsIn[w];
						weightsIn[w] += -learningRate * delta * inPtr[in];
						in++;
					}

					if (!layer.independentInputs()) {
						in = 0;
					}
				}
			}
		}

	public:
		BackpropagationTrainer(double learnRate = 0.1, double error = 0.002, int epochs = 1000)
			: SupervisedNetworkTrainer(learnRate, error, epochs) { }
	};
}