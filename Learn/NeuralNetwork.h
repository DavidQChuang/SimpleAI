#pragma once

#include <vector>
#include <string>

#include "nn/NeuronLayer.h"

namespace nn {
	class NeuralNetwork {
	public:
		typedef INeuronLayer Layer;

	private:
		typedef vector<std::unique_ptr<Layer>> Layers;

		// fields
		Layers nnLayers;

		int ioBufferSize;

		int inputs = 0;
		int outputs = 0;
		
		NeuralNetwork() { }

	public:
		static NeuralNetwork MakeNetwork(initializer_list<Layer*> layerArgs) {
			return NeuralNetwork(layerArgs);
		};

		NeuralNetwork(const NeuralNetwork& net) {
			nnLayers = Layers();

			for (const std::unique_ptr<Layer>& layer : net.nnLayers) {
				Layer* layerCpy = layer.get()->clone();

				nnLayers.push_back(std::unique_ptr<Layer>(layerCpy));
			}

			ioBufferSize = net.ioBufferSize;
			inputs = net.inputs;
			outputs = net.outputs;
		}
		~NeuralNetwork() = default;

		NeuralNetwork(std::initializer_list<Layer*> layerArgs) {
			nnLayers = Layers();

			for (Layer* layer : layerArgs) {
				nnLayers.push_back(std::move(std::unique_ptr<Layer>(layer)));
			}

			if (nnLayers.size() == 0) {
				throw invalid_argument("The neural network cannot have zero layers.");
			}

			int layers = nnLayers.size();
			int last = layers - 1;
			int first = 0;

			int inputsPerNeuron = 1;
			for (int i = 0; i < layers; i++) {
				Layer& layer = *nnLayers[i];

				bool indepInputs = i == first;
				bool useInputs = true; //(i != lastL);
				bool useOutputs = false; //(i != firstL);

				layer.init(inputsPerNeuron, 1, indepInputs, useInputs, useOutputs);

				if (inputsPerNeuron * (layer.size() * indepInputs + 1 - indepInputs) != layer.expectedInputs())
					throw out_of_range("Layers have incompatible in/out sizes.");

				inputsPerNeuron = layer.expectedOutputs();
				ioBufferSize += layer.expectedInputs();
			}

			inputs = (*nnLayers[0]).expectedInputs();
			outputs = (*nnLayers[nnLayers.size() - 1]).expectedOutputs();

			ioBufferSize += outputs;
		}

	public:
		inline int expectedInputs() const { return inputs; }
		inline int expectedOutputs() const { return outputs; }

		inline int expectedBufferSize() const { return ioBufferSize; }

		inline Layer& getLayer(int i) {
			return *nnLayers[i].get();
		}

		inline int depth() const {
			return nnLayers.size();
		}

		double* execute(double* inputs, size_t inLength) const {
			double* buffer = new double[ioBufferSize];
			memcpy(buffer, inputs, inLength * sizeof(double));

			return executeToIOArray(buffer, inLength, ioBufferSize);
		}

		double* executeToIOArray(double* buffer, size_t inLength, size_t bufferSize) const {
			if (inLength != (*nnLayers[0]).expectedInputs())
				throw invalid_argument("Expected input size did not match given input size.");

			if (ioBufferSize != bufferSize)
				throw invalid_argument("Expected buffer size did not match given buffer size.");

			double* inPtr = buffer;
			for (int i = 0; i < nnLayers.size(); i++) {
				Layer& layer = *nnLayers[i];
				int inLen = layer.expectedInputs();
				int outLen = layer.expectedOutputs();

				double* outPtr = inPtr + inLen;

				layer.execute(inPtr, inLen, outPtr, outLen);
				inPtr = outPtr;
			}

			return buffer + (ioBufferSize - (*nnLayers.back()).expectedOutputs());
		}

		void display() {
			printf("\nExpected in/out: %s/%s\n", to_string(inputs).c_str(), to_string(outputs).c_str());

			if (nnLayers.size() == 1) {
				printf("### Input Layer");
				(*nnLayers[0]).display();
			}
			else if (nnLayers.size() >= 2) {
				Layers::iterator it;

				printf("### Input Layer");
				(*nnLayers.front()).display();
				printf("\n");

				for (it = nnLayers.begin() + 1; it < nnLayers.end() - 1; it++) {
					Layer& layer = **it;

					printf("\n### Hidden Layer - %s", layer.name().c_str());
					layer.display();
					printf("\n");
				}

				printf("\n### Output Layer");
				(*nnLayers.back()).display();
			}
		}

		void displayChange(NeuralNetwork other) {
			printf("Expected in/out: %s/%s\n", to_string(inputs).c_str(), to_string(outputs).c_str());

			if (nnLayers.size() == 1) {
				printf("### Input Layer");
				(*nnLayers[0]).display();
				(*other.nnLayers[0]).display();
			}
			else if (nnLayers.size() >= 2) {
				Layers::iterator it, it2;

				printf("### Input Layer");
				(*nnLayers.front()).display();
				(*other.nnLayers.front()).display();
				printf("\n");

				for (it = nnLayers.begin() + 1, it2 = other.nnLayers.begin() + 1; it < nnLayers.end() - 1; it++) {
					Layer& layer1 = **it;
					Layer& layer2 = **it;

					printf("\n### Hidden Layer - %s", layer1.name().c_str());
					layer1.display();
					printf("\n### Hidden Layer - %s", layer2.name().c_str());
					layer2.display();
					printf("\n");
				}

				printf("\n### Output Layer");
				(*nnLayers.back()).display();
				(*other.nnLayers.back()).display();
			}
		}
	};
}