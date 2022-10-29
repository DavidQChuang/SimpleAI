#pragma once

#include <vector>
#include <string>

#include "nn/NeuronLayer.h"

using namespace std;

namespace nn {
	class NeuralNetwork {
	private:
		vector<NeuronLayer> nnLayers;

		int ioBufferSize;

		int inputs = 0;
		int outputs = 0;

		NeuralNetwork() {
			nnLayers = vector<NeuronLayer>();
		}

	public:
		NeuralNetwork(initializer_list<NeuronLayer> getLayers) {
			nnLayers = getLayers;

			if (nnLayers.size() == 0) {
				throw invalid_argument("The neural network cannot have zero layers.");
			}

			int layers = nnLayers.size();
			int lastL = layers - 1;
			int firstL = 0;

			int inputsPerNeuron = 1;
			for (int i = 0; i < layers; i++) {
				NeuronLayer layer = nnLayers[i];

				if (inputsPerNeuron != layer.expectedInputs())
					throw out_of_range("Layers have incompatible I/O buffer sizes.");
				
				bool useInputs = i != lastL;
				bool useOutputs = i != firstL;

				layer.init(inputsPerNeuron, 1, useInputs, useOutputs);

				inputsPerNeuron = layer.expectedOutputs();
				ioBufferSize += layer.expectedInputs();
			}

			inputs = nnLayers[0].expectedInputs();
			outputs = nnLayers[nnLayers.size() - 1].expectedOutputs();

			ioBufferSize += outputs;
		}

		inline int expectedInputs() { return inputs; }
		inline int expectedOutputs() { return outputs; }

		inline int expectedBufferSize() { return ioBufferSize; }

		inline vector<NeuronLayer>& getLayers() { return nnLayers; }

		double* execute(double* inputs, size_t inLength) {
			double* buffer = new double[ioBufferSize];
			memcpy(buffer, inputs, inLength * sizeof(double));

			return executeToIOArray(buffer, inLength, ioBufferSize);
		}

		double* executeToIOArray(double* buffer, size_t inLength, size_t bufferSize) {
			if (inLength != nnLayers[0].expectedInputs())
				throw invalid_argument("Expected input size did not match given input size.");

			if (ioBufferSize != bufferSize)
				throw invalid_argument("Expected buffer size did not match given buffer size.");

			double* inPtr = buffer;
			for (int i = 0; i < nnLayers.size(); i++) {
				NeuronLayer layer = nnLayers[i];
				int inLen = layer.expectedInputs();
				int outLen = layer.expectedOutputs();

				double* outPtr = inPtr + inLen;

				layer.execute(inPtr, inLen, outPtr, outLen);
				inPtr = outPtr;
			}

			return buffer + (ioBufferSize - nnLayers.back().expectedOutputs());
		}

		void display() {
			printf("Expected in/out: %s/%s\n", to_string(inputs).c_str(), to_string(outputs).c_str());

			if (nnLayers.size() == 1) {
				printf("### INPUT LAYER ###\n");
				nnLayers[0].display();
			}
			else if (nnLayers.size() >= 2) {
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

		void displayChange(NeuralNetwork other) {
			printf("Expected in/out: %s/%s\n", to_string(inputs).c_str(), to_string(outputs).c_str());

			if (nnLayers.size() == 1) {
				printf("### INPUT LAYER ###\n");
				nnLayers[0].display();
				other.nnLayers[0].display();
			}
			else if (nnLayers.size() >= 2) {
				vector<NeuronLayer>::iterator it, it2;

				printf("### INPUT LAYER ###\n");
				nnLayers.front().display();
				other.nnLayers.front().display();

				for (it = nnLayers.begin() + 1, it2 = other.nnLayers.begin() + 1; it < nnLayers.end() - 1; it++) {
					NeuronLayer layer = *it;

					printf("### HIDDEN LAYER ### - %s\n", layer.name().c_str());
					(*it).display();
					(*it2).display();
				}

				printf("### OUTPUT LAYER ###\n");
				nnLayers.back().display();
				other.nnLayers.back().display();
			}
		}
	};
}