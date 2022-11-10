#pragma once

#include "SupervisedTrainer.h"

namespace nn {
	class BackpropagationTrainer : public SupervisedTrainer {
	protected:
		/*void initTrainingSet(NeuralNetwork& network,
			double* inputs, size_t inLength, double* expOutputs, size_t outLength) override {
			SupervisedTrainer::initTrainingSet(network, inputs, inLength, expOutputs, outLength);
		}*/

		void trainOnSet(NeuralNetwork& network, double* inputs, double* expOutputs, double* buffer, double* outPtr) {
			vector<double> layerDelta;
			vector<double> oldLayerDelta;
			vector<NeuronLayer>& layers = network.getLayers();

			// Calculate target vs. nn output errors and store them in the layerDelta buffer.
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

				layerDelta.push_back(sumExp - sumNN);
			}

			double* inPtr = outPtr;

			// Update layer weights from back to front.
			for (int l = layers.size() - 1; l >= 0; l--) {
				NeuronLayer& layer = layers[l];
				vector<double>& weightsIn = layer.weightsIn();

				inPtr -= layer.expectedInputs();

				int inputCount = layer.inputsPerNeuron();
				int in = 0;

				// Store current layer deltas and reserve and clear the next layer to 0.
				oldLayerDelta = layerDelta;
				layerDelta.reserve(inputCount);
				for (int i = 0; i < inputCount; i++) {
					if (i >= layerDelta.size()) {
						layerDelta.push_back(0);
					}
					else {
						layerDelta[i] = 0;
					}
				}

				for (int n = 0; n < layer.size(); n++) {
					double weightedSum = 0;

					// Sum weighted inputs of this layer - this is used later
					for (int i = 0; i < inputCount; i++) {
						int w = n * inputCount + i;

						weightedSum += inPtr[in] * weightsIn[w];
						in++;
					}
					in -= inputCount;

					// The delta for this neuron will have been calculated previously -
					// error for output layer, sum of deltas for hidden/input layers,
					// and is then multiplied by f'(h), where h is the weighted sum of inputs.
					double delta = oldLayerDelta[n] * layer.derivActivationFunc(weightedSum);

					// Adjust the weights for each neuron in sequence.
					for (int i = 0; i < inputCount; i++) {
						int w = n * inputCount + i;

						// Each input corresponds to a neuron in the preceding layer.
						// The next layer's delta for that neuron [i] is the sum of this
						// layer's neurons' deltas dj * the weight wij connecting the two
						// neurons for each neuron [j] in this layer.
						layerDelta[i] += delta * weightsIn[w];
						weightsIn[w] += learningRate * delta * inPtr[in];
						in++;
					}

					// If the inputs for this layer's neurons are independent,
					// the inputs overlap instead of being stored sequentially.
					if (!layer.independentInputs()) {
						in = 0;
					}
				}
			}
		}

	public:
		BackpropagationTrainer(double learnRate = 0.1, double error = 0.002, int epochs = 1000)
			: SupervisedTrainer(learnRate, error, epochs) { }
	};
}