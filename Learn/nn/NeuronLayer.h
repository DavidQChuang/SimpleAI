#pragma once

#include <vector>
#include "Neuron.h"

namespace nn {
	class NeuronLayer {
	private:
		vector<Neuron> neurons;

		int inputsPerNeuron; // input per neuron
		int outputsPerNeuron; // output per neuron

		int neuronCount;
		string layerName;

	private:
		inline void initNeurons(ActivationFunction func) {
			neurons.clear();

			for (int i = 0; i < neuronCount; i++) {
				vector<double> inputWeights;
				for (int i = 0; i < inputsPerNeuron; i++) {
					inputWeights.push_back(0);
				}

				vector<double> outputWeights;
				for (int i = 0; i < outputsPerNeuron; i++) {
					outputWeights.push_back(0);
				}

				neurons.push_back(Neuron(func, inputWeights, outputWeights));
			}
		}

	public:
		NeuronLayer(int count, string name)
			: neuronCount(count), layerName(name) {
			inputsPerNeuron = 0;
			outputsPerNeuron = 0;
		}

		void init(NeuronLayer* prev, NeuronLayer* next, ActivationFunction func) {
			if (prev == NULL)	inputsPerNeuron = 1;
			else				inputsPerNeuron = prev->neuronCount;

			outputsPerNeuron = 1;

			//if (next == NULL)	outputsPerNeuron = 1;
			//else				outputsPerNeuron = next->neuronCount;

			initNeurons(func);
		}

		int expectedInputs() {
			return inputsPerNeuron * neurons.size();
		}

		int expectedOutputs() {
			return inputsPerNeuron * neurons.size();
		}

		void execute(double* input, int inputLength, double* output, int outputLength) {
			if (input == NULL) throw invalid_argument("Input layer received null input pointer.");
			if (output == NULL) throw invalid_argument("Input layer received null output pointer.");

			int neuronCount = neurons.size();

			if (inputLength != inputsPerNeuron * neuronCount) throw invalid_argument("Input buffer length is invalid.");
			if (outputLength != neuronCount) throw invalid_argument("Output buffer length is invalid, must be equal to neuron count.");

			for (int n = 0; n < neuronCount; n++) {
				Neuron neuron = neurons[n];

				double sum = 0;
				double* dataIn = input + n * inputsPerNeuron;
				double* dataOut = output + n * outputsPerNeuron;

				// sum weights * input
				vector<double> weightsIn= neuron.weightsIn();
				for (int i = 0; i < inputsPerNeuron; i++) {
					sum += dataIn[i] * weightsIn[i];
				}

				// activation function on sum
				sum = neuron.activationFunc(sum);

				// copy outputs to buffer
				for (int i = 0; i < outputsPerNeuron; i++) {
					dataOut[i] = sum;
				}
			}
		}

		int size() {
			return neuronCount;
		}

		string name() {
			return layerName;
		}

		void display() {
			for (int i = 0; i < neuronCount; i++) {
				Neuron n = neurons[i];

				printf("\nNeuron #%s\n", to_string(i).c_str());
				n.display();
			}
		}
	};
}