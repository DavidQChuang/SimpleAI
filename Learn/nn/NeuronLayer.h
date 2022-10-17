#pragma once

#include <vector>
#include "Neuron.h"

namespace nn {
	class NeuronLayer {
	private:
		vector<Neuron> neurons;

		int		mInputsPerNeuron; // input per neuron
		int		mOutputsPerNeuron; // output per neuron

		int		neuronCount;
		string	layerName;

	private:
		inline void initNeurons(ActivationFunction func) {
			neurons.clear();

			for (int i = 0; i < neuronCount; i++) {
				vector<double> inputWeights;
				for (int i = 0; i < mInputsPerNeuron; i++) {
					inputWeights.push_back(0);
				}

				vector<double> outputWeights;
				for (int i = 0; i < mOutputsPerNeuron; i++) {
					outputWeights.push_back(0);
				}

				neurons.push_back(Neuron(func, inputWeights, outputWeights));
			}
		}

	public:
		NeuronLayer(int count, string name)
			: neuronCount(count), layerName(name) {
			mInputsPerNeuron = 0;
			mOutputsPerNeuron = 0;

			if (count == 0) throw out_of_range("Invalid neuron count, cannot be zero.");
		}
		NeuronLayer(const NeuronLayer& other) {
			for (int i = 0; i < other.neurons.size(); i++) {
				Neuron neuronCopy = Neuron(other.neurons[i]);
				neurons.push_back(neuronCopy);
			}

			mInputsPerNeuron = other.mInputsPerNeuron;
			mOutputsPerNeuron = other.mOutputsPerNeuron;

			neuronCount = other.neuronCount;
			layerName = other.layerName;
		}

		void init(NeuronLayer* prev, NeuronLayer* next, ActivationFunction func) {
			if (prev == NULL)	mInputsPerNeuron = 1;
			else				mInputsPerNeuron = prev->neuronCount;

			mOutputsPerNeuron = 1;

			//if (next == NULL)	outputsPerNeuron = 1;
			//else				outputsPerNeuron = next->neuronCount;

			initNeurons(func);
		}

		size_t expectedInputs() {
			return mInputsPerNeuron * neurons.size();
		}

		size_t expectedOutputs() {
			return mInputsPerNeuron * neurons.size();
		}

		int inputPerNeuron() { return mInputsPerNeuron; }

		int outputsPerNeuron() { return mOutputsPerNeuron; }

		vector<Neuron> getNeurons() {
			return vector<Neuron>(neurons);
		}

		void execute(double* input, int inputLength, double* output, int outputLength) {
			if (mInputsPerNeuron == 0 || mOutputsPerNeuron == 0) throw out_of_range("Invalid input/output count, cannot be zero. Likely did not init.");

			if (input == NULL) throw invalid_argument("Input layer received null input pointer.");
			if (output == NULL) throw invalid_argument("Input layer received null output pointer.");

			size_t neuronCount = neurons.size();

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
			for (int i = 0; i < neuronCount; i++) {
				Neuron n = neurons[i];

				printf("\nNeuron #%s\n", to_string(i).c_str());
				n.display();
			}
		}
	};
}