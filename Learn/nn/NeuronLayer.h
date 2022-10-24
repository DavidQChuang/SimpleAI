#pragma once

#include <vector>
#include "Neuron.h"
#include <random>

namespace nn {
	class NeuronLayer {
	private:
		Neuron* neurons = nullptr;

		int		mInputsPerNeuron = 0; // input per neuron
		int		mOutputsPerNeuron = 0; // output per neuron

		int		neuronCount = 0;
		string	layerName = "";

	public:
		NeuronLayer(int count, string name)
			: neuronCount(count), layerName(name) {
			if (count == 0) throw out_of_range("Invalid neuron count, cannot be zero.");
		}

		void init(NeuronLayer* prev, NeuronLayer* next, ActivationFunction func, Neuron* neuronBuf) {
			if (prev == NULL)	mInputsPerNeuron = 1;
			else				mInputsPerNeuron = prev->neuronCount;

			mOutputsPerNeuron = 1;

			//if (next == NULL)	outputsPerNeuron = 1;
			//else				outputsPerNeuron = next->neuronCount;

			neurons = neuronBuf;
			for (int n = 0; n < neuronCount; n++) {
				vector<double> inputWeights;
				for (int i = 0; i < mInputsPerNeuron; i++) {
					inputWeights.push_back((double)rand() / RAND_MAX);
				}

				vector<double> outputWeights;
				for (int i = 0; i < mOutputsPerNeuron; i++) {
					outputWeights.push_back((double)rand() / RAND_MAX);
				}

				neurons[n] = Neuron(func, inputWeights, outputWeights);
			}
		}

		int expectedInputs() {
			return mInputsPerNeuron * neuronCount;
		}

		int expectedOutputs() {
			return mOutputsPerNeuron * neuronCount;
		}

		Neuron* getNeurons() {
			return neurons;
		}

		void execute(double* input, int inputLength, double* output, int outputLength) {
			if (neurons == nullptr) throw invalid_argument("Uninitialized layer.");
			if (mInputsPerNeuron == 0 || mOutputsPerNeuron == 0) throw out_of_range("Invalid input/output count, cannot be zero. Likely did not init.");
			

			if (input == NULL) throw invalid_argument("Layer received null input pointer.");
			if (output == NULL) throw invalid_argument("Layer received null output pointer.");

			if (inputLength != mInputsPerNeuron * neuronCount) throw invalid_argument("Input buffer length is invalid.");
			if (outputLength != neuronCount) throw invalid_argument("Output buffer length is invalid, must be equal to neuron count.");

			for (int n = 0; n < neuronCount; n++) {
				Neuron neuron = neurons[n];

				double sum = 0;
				double* dataIn = input + n * mInputsPerNeuron;
				double* dataOut = output + n * mOutputsPerNeuron;

				// sum weights * input
				vector<double> weightsIn= neuron.weightsIn();
				for (int i = 0; i < mInputsPerNeuron; i++) {
					sum += dataIn[i] * weightsIn[i];
				}

				// activation function on sum
				sum = neuron.activationFunc(sum);

				// copy outputs to buffer
				for (int i = 0; i < mOutputsPerNeuron; i++) {
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
			if (neurons == nullptr) {
				printf("UNINITIALIZED LAYER OF SIZE %s", to_string(neuronCount).c_str());
				return;
			}

			for (int i = 0; i < neuronCount; i++) {
				Neuron n = neurons[i];

				printf("\nNeuron #%s\n", to_string(i).c_str());
				n.display();
			}
		}
	};
}