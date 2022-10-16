#pragma once

#include <vector>
#include <string>

#include "nn/Neuron.h"
#include "nn/NeuronLayer.h"

using namespace std;

namespace nn {
	class NeuralNetwork {
	private:
		NeuronLayer* input;
		NeuronLayer* output;
		vector<NeuronLayer> hiddenLayers;

	public:
		NeuralNetwork(NeuronLayer* inputLayer, NeuronLayer* outputLayer, vector<NeuronLayer> hiddenLayersList) {
			input = inputLayer;
			output = outputLayer;
			hiddenLayers = hiddenLayersList;

			if (input == NULL || output == NULL) {
				throw invalid_argument("The neural network cannot have null input or output layers.");
			}
		}

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

		double* execute() {
			int count = 0;
			int finalOutputOffset = 0;

			if (input == output) {
				count += input->expectedInputs();
				finalOutputOffset = count;
				count += input->expectedOutputs();
			}
			else {
				int prevOutputs;

				count = count + input->expectedInputs();
				count = count + (prevOutputs = input->expectedOutputs());

				for (int i = 0; i < hiddenLayers.size(); i++) {
					NeuronLayer layer = hiddenLayers[i];

					if (prevOutputs != layer.expectedInputs())
						throw out_of_range("Expected input of layer did not match expected error of previous layer.");

					count = count + (prevOutputs = layer.expectedOutputs());
				}

				if (prevOutputs != output->expectedInputs())
					throw out_of_range("Expected input of output layer did not match expected error of previous layer.");
				
				finalOutputOffset = count;
				count += output->expectedOutputs();
			}

			double* buffer = new double[count];

			if (input == output) {

			}

			return buffer + finalOutputOffset;
		}
	};
}