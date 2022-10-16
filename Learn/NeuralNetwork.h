#pragma once

#include <vector>
#include <string>

#include "nn/Neuron.h"
#include "nn/NeuronLayer.h"

using namespace std;

namespace nn {
	class NeuralNetwork {
	private:
		vector<NeuronLayer> nnLayers;
		int ioBufferSize;
		int expectedInputs;
		int expectedOutputs;

	public:
		NeuralNetwork(vector<NeuronLayer> layers) {
			nnLayers = layers;

			if (nnLayers.size() == 0) {
				throw invalid_argument("The neural network cannot have zero layers.");
			}
		}

		void init(ActivationFunction func) {
			int bufSize = 0;

			NeuronLayer input = nnLayers.front();
			NeuronLayer output = nnLayers.back();

			if (nnLayers.size() == 1) {
				// init layer
				nnLayers.front().init(NULL, NULL, func);

				// calculate output buffer size
				bufSize += input.expectedInputs();
				bufSize += input.expectedOutputs();
			}
			else if(nnLayers.size() > 2) {
				vector<NeuronLayer>::iterator it;

				NeuronLayer input = nnLayers.front();
				NeuronLayer output = nnLayers.back();
				NeuronLayer* prev = &input;

				// init all layers
				input.init(NULL, &nnLayers[1], func);
				for (it = nnLayers.begin() + 1; it < nnLayers.end() - 1; it++) {
					NeuronLayer layer = *it;
					layer.init(prev, &*(it + 1), func);
					prev = &layer;
				}
				output.init(&*it, NULL, func);

				// calculate output buffer size
				int prevOutputs;
				vector<NeuronLayer>::iterator it;

				bufSize = bufSize + input.expectedInputs();
				bufSize = bufSize + (prevOutputs = input.expectedOutputs());

				for (it = nnLayers.begin() + 1; it < nnLayers.end() - 1; it++) {
					NeuronLayer layer = *it;

					if (prevOutputs != layer.expectedInputs())
						throw out_of_range("Expected input of layer did not match expected error of previous layer.");

					bufSize = bufSize + (prevOutputs = layer.expectedOutputs());
				}

				if (prevOutputs != output.expectedInputs())
					throw out_of_range("Expected input of output layer did not match expected error of previous layer.");

				bufSize += output.expectedOutputs();
			}

			expectedInputs = nnLayers.front().expectedInputs();
			expectedOutputs = nnLayers.back().expectedOutputs();
			ioBufferSize = bufSize;
		}

		void display() {
			if (nnLayers.size() == 1) {
				printf("### INPUT LAYER ###\n");
				nnLayers[0].display();
			}
			else if (nnLayers.size() > 2) {
				vector<NeuronLayer>::iterator it;

				printf("### INPUT LAYER ###\n");
				nnLayers.front().display();
				for (it = nnLayers.begin() + 1; it < nnLayers.end() - 1; it++) {
					NeuronLayer layer = *it;

					printf("### HIDDEN LAYER ### - %s\n", layer.name().c_str());
					layer.display();
				}

				printf("### OUTPUT LAYER ###\n");
				nnLayers.back().display();
			}
		}

		int expectedInputs() {
			return nnLayers.front().expectedInputs();
		}

		int expectedOutputs() {
			return nnLayers.back().expectedOutputs();
		}

		double* execute() {
			double* buffer = new double[ioBufferSize];
			double* inPtr = buffer;

			int remaining = ioBufferSize;
			int lastOutLen = 0;

			for (int i = 0; i < nnLayers.size(); i++) {
				NeuronLayer layer = nnLayers[i];
				int inLen = layer.expectedInputs();
				int outLen = layer.expectedOutputs();

				remaining -= inLen;
				if (inLen < 0) throw out_of_range("Exceeded buffer capacity.");

				layer.execute(inPtr, inLen, inPtr + outLen, outLen);
				lastOutLen = outLen;
			}

			return buffer + (ioBufferSize - lastOutLen);
		}
	};
}