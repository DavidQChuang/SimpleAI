#pragma once

#include <vector>
#include <string>

#include "nn/NeuronLayer.h"

using namespace std;

namespace nn {
	// Feedforward 
	template<typename... Layers>
	class NeuralNetwork {
	private:
		tuple<Layers...> nnLayers;

		int ioBufferSize;

		int inputs = 0;
		int outputs = 0;

		NeuralNetwork() {
		}

#define NNLAYERS_SIZE sizeof...(Layers)
#define NNLAYERS(i) std::get<i>(nnLayers)
#define NNLAYERS_FRONT std::get<0>(nnLayers)
#define NNLAYERS_BACK std::get<NNLAYERS_SIZE - 1>(nnLayers)

	public:
		NeuralNetwork(Layers... layers) {
			nnLayers = make_tuple(layers);

			if (NNLAYERS_SIZE == 0) {
				throw invalid_argument("The neural network cannot have zero layers.");
			}

			int layers = NNLAYERS_SIZE;
			int last = layers - 1;
			int first = 0;

			int inputsPerNeuron = 1;
			for (int i = 0; i < layers; i++) {
				NeuronLayer& layer = NNLAYERS(i);
				
				bool indepInputs = i == first;
				bool useInputs = true; //(i != lastL);
				bool useOutputs = false; //(i != firstL);

				layer.init(inputsPerNeuron, 1, indepInputs, useInputs, useOutputs);

				if (inputsPerNeuron * (layer.size() * indepInputs + 1 - indepInputs) != layer.expectedInputs())
					throw out_of_range("Layers have incompatible in/out sizes.");

				inputsPerNeuron = layer.expectedOutputs();
				ioBufferSize += layer.expectedInputs();
			}

			inputs = NNLAYERS(0).expectedInputs();
			outputs = NNLAYERS(NNLAYERS_SIZE - 1).expectedOutputs();

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
			if (inLength != NNLAYERS(0).expectedInputs())
				throw invalid_argument("Expected input size did not match given input size.");

			if (ioBufferSize != bufferSize)
				throw invalid_argument("Expected buffer size did not match given buffer size.");

			double* inPtr = buffer;
			for (int i = 0; i < NNLAYERS_SIZE; i++) {
				NeuronLayer& layer = NNLAYERS(i);
				int inLen = layer.expectedInputs();
				int outLen = layer.expectedOutputs();

				double* outPtr = inPtr + inLen;

				layer.execute(inPtr, inLen, outPtr, outLen);
				inPtr = outPtr;
			}

			return buffer + (ioBufferSize - nnLayers.back().expectedOutputs());
		}

		void display() {
			printf("\nExpected in/out: %s/%s\n", to_string(inputs).c_str(), to_string(outputs).c_str());

			if (NNLAYERS_SIZE == 1) {
				printf("### Input Layer");
				NNLAYERS(0).display();
			}
			else if (NNLAYERS_SIZE >= 2) {
				printf("### Input Layer");
				NNLAYERS_FRONT.display();
				printf("\n");

				for (int i = 1; i < NNLAYERS_SIZE - 1; i++) {
					NeuronLayer layer = NNLAYERS(i);

					printf("\n### Hidden Layer - %s", layer.name().c_str());
					layer.display();
					printf("\n");
				}

				printf("\n### Output Layer");
				NNLAYERS_BACK.display();
			}
		}

#define OTHER_NNLAYERS(i) std::get<i>(other.nnLayers)
#define OTHER_NNLAYERS_FRONT std::get<0>(other.nnLayers)
#define OTHER_NNLAYERS_BACK std::get<NNLAYERS_SIZE - 1>(other.nnLayers)
		void displayChange(NeuralNetwork other) {
			printf("Expected in/out: %s/%s\n", to_string(inputs).c_str(), to_string(outputs).c_str());

			if (NNLAYERS_SIZE == 1) {
				printf("### Input Layer");
				NNLAYERS(0).display();
				OTHER_NNLAYERS(0).display();
			}
			else if (NNLAYERS_SIZE >= 2) {
				printf("### Input Layer");
				NNLAYERS_FRONT.display();
				OTHER_NNLAYERS_FRONT.display();
				printf("\n");

				for (int i = 1; i < NNLAYERS_SIZE - 1; i++) {
					NeuronLayer layer = NNLAYERS(i);
					NeuronLayer layer2 = OTHER_NNLAYERS(i);

					printf("\n### Hidden Layer - %s", layer.name().c_str());
					layer.display();
					layer2.display();
					printf("\n");
				}

				printf("\n### Output Layer");
				NNLAYERS_BACK.display();
				OTHER_NNLAYERS_BACK.display();
			}
		}
	};

	template<typename... Layers>
	NeuralNetwork<Layers...> MakeNetwork(Layers... layers) { return NeuralNetwork(layers); }
}