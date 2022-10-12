#pragma once

#include <vector>
#include <string>

#include "nn/Neuron.h"
#include "nn/NeuralNetworkTrainer.h"

using namespace std;

namespace nn {
	class NeuralNetwork {
	private:
		NeuronLayer* input;
		NeuronLayer* output;
		vector<NeuronLayer> hiddenLayers;

	public:
		NeuralNetwork(NeuronLayer* input, NeuronLayer* output, vector<NeuronLayer> hiddenLayers)
			: input(input), output(output), hiddenLayers(hiddenLayers) { }

		void init(ActivationFunction func) {
			if (input == output) {
				input->init(NULL, NULL, func);
			}
			else if(hiddenLayers.size() == 0) {
				input->init(NULL, output, func);
				output->init(input, NULL, func);
			}
			else{
				vector<NeuronLayer>::iterator it;
				NeuronLayer* prev = input;
				NeuronLayer* last;

				input->init(NULL, &hiddenLayers.front(), func);

				for (it = hiddenLayers.begin();;) {
					NeuronLayer layer = *it++;
					NeuronLayer next = *it;

					if (it >= hiddenLayers.end()) {
						layer.init(prev, output, func);
						prev = &layer;
						break;
					}

					layer.init(prev, &next, func);
					prev = &layer;
				}

				output->init(prev, NULL, func);
			}
		}

		void display() {
			printf("### INPUT LAYER ###\n");
			input->display();

			vector<NeuronLayer>::iterator it;
			for (it = hiddenLayers.begin(); it < hiddenLayers.end(); it++) {
				NeuronLayer layer = (*it);

				printf("### HIDDEN LAYER ### - %s\n", layer.name().c_str());
				layer.display();
			}

			printf("### OUTPUT LAYER ###\n");
			output->display();
		}

		NeuralNetwork train(NetworkTrainer trainer) {

			return *this;
		}
	};
	
	class NeuronLayer {
	private:
		vector<Neuron> neurons;

		int inputPerNeuron; // input per neuron
		int outputPerNeuron; // output per neuron

		int neuronCount;
		string layerName;

	public:
		NeuronLayer(int count, string name)
			: neuronCount(count), layerName(name) {}

		void init(NeuronLayer* prev, NeuronLayer* next, ActivationFunction func) {
			if (prev == NULL)	inputPerNeuron = 1;
			else				inputPerNeuron = prev->neuronCount;

			if (next == NULL)	outputPerNeuron = 1;
			else				outputPerNeuron = prev->neuronCount;

			for (int i = 0; i < neuronCount; i++) {
				vector<double> inputWeights;
				for (int i = 0; i < inputPerNeuron; i++) {
					inputWeights.push_back(0);
				}

				vector<double> outputWeights;
				for (int i = 0; i < outputPerNeuron; i++) {
					outputWeights.push_back(0);
				}

				neurons.push_back(Neuron(func, inputWeights, outputWeights));
			}
		}

		int neuronCount() {
			return neuronCount;
		}

		string name() {
			return layerName;
		}

		void display() {
			for (int i = 0; i < neuronCount; i++) {
				Neuron n = neurons[i];

				printf("\nNeuron #%s\n", i);
				n.display();
			}
		}
	};
}