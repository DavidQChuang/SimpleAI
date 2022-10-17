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
		size_t ioBufferSize;
		int inputs;
		int outputs;

	public:
		NeuralNetwork(vector<NeuronLayer> getLayers) {
			nnLayers = getLayers;

			if (nnLayers.size() == 0) {
				throw invalid_argument("The neural network cannot have zero layers.");
			}
		}
		NeuralNetwork(const NeuralNetwork& other) {
			for (int i = 0; i < other.nnLayers.size(); i++) {
				NeuronLayer layerCopy = NeuronLayer(other.nnLayers[i]);
				nnLayers.push_back(layerCopy);
			}

			ioBufferSize = other.ioBufferSize;
			inputs = other.inputs;
			outputs = other.outputs;
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
				int prevOutputs;

				// calculate output buffer size
				bufSize = bufSize + input.expectedInputs();
				bufSize = bufSize + (prevOutputs = input.expectedOutputs());

				// init all layers
				input.init(NULL, &nnLayers[1], func);
				for (it = nnLayers.begin() + 1; it < nnLayers.end() - 1; it++) {
					NeuronLayer layer = *it;
					layer.init(prev, &*(it + 1), func);
					prev = &layer;

					if (prevOutputs != layer.expectedInputs())
						throw out_of_range("Expected input of layer did not match expected error of previous layer.");

					bufSize = bufSize + (prevOutputs = layer.expectedOutputs());
				}
				output.init(&*it, NULL, func);

				if (prevOutputs != output.expectedInputs())
					throw out_of_range("Expected input of output layer did not match expected error of previous layer.");

				bufSize += output.expectedOutputs();
			}

			inputs = nnLayers.front().expectedInputs();
			outputs = nnLayers.back().expectedOutputs();
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
			return inputs;
		}

		int expectedOutputs() {
			return outputs;
		}

		size_t expectedBufferSize() {
			return ioBufferSize;
		}

		vector<NeuronLayer> getLayers() {
			return vector<NeuronLayer>(nnLayers);
		}

		double* execute(double* inputs, size_t inLength) {
			double* buffer = new double[ioBufferSize];
			memcpy(buffer, inputs, inLength);

			return executeToIOArray(buffer, inLength, ioBufferSize);
		}

		double* executeToIOArray(double* buffer, size_t inLength, size_t bufferSize) {
			double* inPtr = buffer;

			int remaining = ioBufferSize;
			int lastOutLen = 0;

			if (inLength != nnLayers[0].expectedInputs())
				throw invalid_argument("Expected input size did not match given input size.");

			if (ioBufferSize != bufferSize)
				throw invalid_argument("Expected buffer size did not match given buffer size.");

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