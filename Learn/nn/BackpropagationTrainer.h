#pragma once

#include "SupervisedTrainer.h"
#include <tuple>

namespace nn {
	template<typename... LayerArgs>
	class BackpropagationTrainer : public SupervisedTrainer<LayerArgs...> {
	private:
		double momentum;
		vector<double> prevWeightDeltas;

	protected:
		void initTraining(FFNeuralNetwork<LayerArgs...>& network,
			int trainingSets,
			double** inputSet, size_t inLength,
			double** expOutputSet, size_t outLength) override {
			for (int l = 0; l < network.depth(); l++) {
				NeuralNetwork::Layer& layer = network.getLayer(l);
				for (int n = 0; n < layer.size(); n++) {
					for (int i = 0; i < layer.inputsPerNeuron(); i++) {
						prevWeightDeltas.push_back(0);
					}
				}
			}
		}

		void trainOnSet(FFNeuralNetwork<LayerArgs...>& network, double* inputs, double* expOutputs, double* buffer, double* outPtr) override {
			vector<double> layerDelta;
			vector<double> oldLayerDelta;

			// Calculate target vs. nn output errors and store them in the layerDelta buffer.
			int out = 0;
			NeuralNetwork::Layer& outputLayer = network.getLayer(network.depth() - 1);

			bool isSoftmax = dynamic_cast<FFVNeuronLayer<VectorFunc::Softmax>*>(&outputLayer);
			for (int n = 0; n < outputLayer.size(); n++) {
				double y = 0;
				double t = 0;

				for (int o = 0; o < outputLayer.outputsPerNeuron(); o++) {
					y += outPtr[out];
					t += expOutputs[out];
					out++;
				}

				if (isSoftmax) {
					layerDelta.push_back(t / (y + 1e-7));
				}
				else {
					layerDelta.push_back(t - y);
				}
			}

			double* inPtr = outPtr;

			// Update layer weights from back to front.
			int wd = 0;
			for (int l = network.depth() - 1; l >= 0; l--) {
				NeuralNetwork::Layer& layer = network.getLayer(l);
				vector<double>& weightsIn = layer.weightsIn();

				inPtr -= layer.totalInputs();

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
					double delta = oldLayerDelta[n] * layer.derivActivationFunc(weightedSum, n);

					// Adjust the weights for each neuron in sequence.
					for (int i = 0; i < inputCount; i++) {
						int w = n * inputCount + i;

						double weightDelta = this->learningRate * delta * inPtr[in] + momentum * prevWeightDeltas[wd];

						// Each input corresponds to a neuron in the preceding layer.
						// The next layer's delta for that neuron [i] is the sum of this
						// layer's neurons' deltas dj * the weight wij connecting the two
						// neurons for each neuron [j] in this layer.
						layerDelta[i] += delta * weightsIn[w];
						if(layer.useInputs())
							weightsIn[w] += weightDelta;

						prevWeightDeltas[wd] = weightDelta;

						in++;
						wd++;
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
		BackpropagationTrainer(double learnRate = 0.1, double error = 0.002, int epochs = 1000, double momentum = 0.5)
			: SupervisedTrainer<LayerArgs...>(learnRate, error, epochs), momentum(momentum){ }
	};
}