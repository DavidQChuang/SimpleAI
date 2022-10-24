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

		Neuron* neuronBuf = nullptr;
		size_t ioBufferSize = 0;

		int inputs = 0;
		int outputs = 0;

		NeuralNetwork() {
			nnLayers = vector<NeuronLayer>();
		}

	public:
		NeuralNetwork(vector<NeuronLayer> getLayers) {
			nnLayers = getLayers;

			if (nnLayers.size() == 0) {
				throw invalid_argument("The neural network cannot have zero layers.");
			}
		}
		~NeuralNetwork() {
			delete[] neuronBuf;
		}

		static NeuralNetwork copy(const NeuralNetwork& other) {
			NeuralNetwork nn = NeuralNetwork();

			for (int i = 0; i < other.nnLayers.size(); i++) {
				NeuronLayer layer = other.nnLayers[i];
				NeuronLayer layerCopy = NeuronLayer(layer.size(), layer.name());
				nn.nnLayers.push_back(layerCopy);
			}

			nn.ioBufferSize = other.ioBufferSize;
			if (other.neuronBuf != nullptr) {
				nn.neuronBuf = new Neuron[nn.ioBufferSize];
				memcpy(nn.neuronBuf, other.neuronBuf, nn.ioBufferSize);
			}

			nn.inputs = other.inputs;
			nn.outputs = other.outputs;

			return nn;
		}

		void init(ActivationFunction func) {
			int bufSize = 0;
			int nBufSize = 0;

			NeuronLayer input = nnLayers.front();
			NeuronLayer output = nnLayers.back();

			if (nnLayers.size() == 1) {
				// calculate output buffer size
				bufSize += input.expectedInputs();
				bufSize += input.expectedOutputs();
				nBufSize += input.size();

				// init buffer
				neuronBuf = new Neuron[nBufSize];

				// init layer
				nnLayers.front().init(NULL, NULL, func, neuronBuf);
			}
			else if(nnLayers.size() >= 2) {
				vector<NeuronLayer>::iterator it;

				// calculate output buffer size
				int prevOutputs;

				bufSize = bufSize + input.expectedInputs();
				bufSize = bufSize + (prevOutputs = input.expectedOutputs());
				nBufSize += input.size();

				for (it = nnLayers.begin() + 1; it < nnLayers.end() - 1; it++) {
					NeuronLayer layer = *it;

					if (prevOutputs != layer.expectedInputs())
						throw out_of_range("Expected input of layer did not match expected output of previous layer.");

					bufSize = bufSize + (prevOutputs = layer.expectedOutputs());
					nBufSize += layer.size();
				}

				if (prevOutputs != output.expectedInputs())
					throw out_of_range("Expected input of output layer did not match expected output of previous layer.");

				bufSize += output.expectedOutputs();
				nBufSize += output.size();

				// init buffer
				neuronBuf = new Neuron[nBufSize];

				// init all layers
				NeuronLayer prev = input;
				Neuron* layerNeurons = neuronBuf;

				input.init(NULL, &nnLayers[1], func, layerNeurons);
				layerNeurons += input.size();
				for (it = nnLayers.begin() + 1; it < nnLayers.end() - 1; it++) {
					NeuronLayer layer = *it;
					NeuronLayer next = *(it + 1);

					layer.init(&prev, &next, func, layerNeurons);

					prev = layer;
					layerNeurons += layer.size();
				}
				output.init(&nnLayers[nnLayers.size() - 2], NULL, func, layerNeurons);
			}

			inputs = nnLayers.front().expectedInputs();
			outputs = nnLayers.back().expectedOutputs();
			ioBufferSize = bufSize;
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
			memcpy(buffer, inputs, inLength * sizeof(double));

			return executeToIOArray(buffer, inLength, ioBufferSize);
		}

		double* executeToIOArray(double* buffer, size_t inLength, size_t bufferSize) {
			int remaining = ioBufferSize;

			if (inLength != nnLayers[0].expectedInputs())
				throw invalid_argument("Expected input size did not match given input size.");

			if (ioBufferSize != bufferSize)
				throw invalid_argument("Expected buffer size did not match given buffer size.");

			double* inPtr = buffer;
			for (int i = 0; i < nnLayers.size(); i++) {
				NeuronLayer layer = nnLayers[i];
				int inLen = layer.expectedInputs();
				int outLen = layer.expectedOutputs();

				remaining -= inLen;
				if (remaining < 0) throw out_of_range("Exceeded buffer capacity.");

				layer.execute(inPtr, inLen, inPtr + inLen, outLen);
				inPtr += inLen;
			}

			return buffer + (ioBufferSize - nnLayers.back().expectedOutputs());
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