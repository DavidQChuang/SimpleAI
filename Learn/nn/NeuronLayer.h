#pragma once

#include <vector>
#include <random>

namespace nn {
	enum ActivationFunction {
		Step, Linear, Siglog, Hypertan
	};

	class NeuronLayer {
	private:
		int		mNeuronInputs = 0; // input per neuron
		int		mNeuronOutputs = 0; // output per neuron
		bool	mUseInputs = true;
		bool	mUseOutputs = true;
		int		neuronCount = 0;

		double* inputWeights = nullptr;
		double* outputWeights = nullptr;

		ActivationFunction func = Step;

		string	layerName = "";

	public:
		NeuronLayer(int count, ActivationFunction f, string name = "Layer") {
			if (count == 0) throw out_of_range("Invalid neuron count, cannot be zero.");

			neuronCount = count;
			func = f;

			layerName = name;
		}

		~NeuronLayer() {
			delete[] inputWeights;
			inputWeights = nullptr;
		}

		void init(int inputsPerNeuron, int outputsPerNeuron, bool useInputs, bool useOutputs) {
			mNeuronInputs = inputsPerNeuron;
			mNeuronOutputs = outputsPerNeuron;

			mUseInputs = useInputs;
			mUseOutputs = useOutputs;

			if (useInputs) {
				int inputs = expectedInputs();
				inputWeights = new double[inputs];
				for (int i = 0; i < inputs; i++) {
					inputWeights[i] = (double)rand() / RAND_MAX;
				}
			}

			if (useOutputs) {
				int outputs = expectedOutputs();
				outputWeights = new double[outputs];
				for (int i = 0; i < outputs; i++) {
					outputWeights[i] = (double)rand() / RAND_MAX;
				}
			}
		}

		inline int expectedInputs() { return mNeuronInputs * neuronCount; }
		inline int expectedOutputs() { return mNeuronOutputs * neuronCount; }

		inline int size() { return neuronCount; }
		inline string name() { return layerName; }

		inline double* weightsIn(int neuron) {
			if (neuron >= neuronCount) throw invalid_argument("Neuron index was out of range.");
			
			return inputWeights + neuron * mNeuronInputs;
		}
		inline double* weightsOut(int neuron) {
			if (neuron >= neuronCount) throw invalid_argument("Neuron index was out of range.");

			return outputWeights + neuron * mNeuronInputs;
		}

		void execute(double* input, int inputLength, double* output, int outputLength) {
			if (inputWeights == nullptr) throw invalid_argument("Uninitialized layer.");

			if (input == NULL) throw invalid_argument("Null input pointer.");
			if (output == NULL) throw invalid_argument("Null output pointer.");

			if (inputLength != expectedInputs()) throw invalid_argument("Input buffer length is invalid.");
			if (outputLength != expectedOutputs()) throw invalid_argument("Output buffer length is invalid.");

			int inWOffset = 0;
			int outWOffset = 0;
			for (int n = 0; n < neuronCount; n++) {
				double sum = 0;

				// sum weights * input
				if (mUseInputs) {
					for (int i = 0; i < mNeuronInputs; i++) {
						sum += input[i] * inputWeights[inWOffset++];
					}
				}
				else {
					for (int i = 0; i < mNeuronInputs; i++) {
						sum += input[i];
					}
				}

				// activation function on sum
				sum = activationFunc(sum);

				// copy outputs to buffer
				if (mUseOutputs) {
					for (int i = 0; i < mNeuronOutputs; i++) {
						output[i] = sum * outputWeights[outWOffset++];
					}
				}
				else {
					for (int i = 0; i < mNeuronOutputs; i++) {
						output[i] = sum;
					}
				}

				input += mNeuronInputs;
				output += mNeuronOutputs;
			}
		}

		void display() {
			if (inputWeights == nullptr) {
				printf("Uninitialized layer: %dx%d inputs, %dx%d outputs",
					mNeuronInputs, neuronCount, mNeuronOutputs, neuronCount);
				return;
			}

			for (int n = 0; n < neuronCount; n++) {
				printf("\nNeuron #%d\n", n);

				if (mUseInputs) {
					printf("Input weights:\n[");
					for (int i = 0; i < mNeuronInputs; i++) {
						printf("%f.3", inputWeights[n * mNeuronInputs + i]);

						if (i + 1 < mNeuronInputs) {
							printf(", ");
						}
						else break;
					}
					printf("]\n");
				}

				if (mUseOutputs) {
					printf("Input weights:\n[");
					for (int i = 0; i < mNeuronOutputs; i++) {
						printf("%f.3", outputWeights[n * mNeuronOutputs + i]);

						if (i + 1 < mNeuronOutputs) {
							printf(", ");
						}
						else break;
					}
					printf("]\n");
				}
			}
		}

		double activationFunc(double v) {
			switch (func) {
			case Step:
				return v < 0 ? 0 : 1;
				break;
			case Linear:
				return v;
			case Siglog:
				return 1 / (1 + exp(-v));
			case Hypertan:
				return (1 - exp(-v)) / (1 + exp(-v));
			}

			throw invalid_argument("There is no function for this activation function type.");
		}

		double derivActivationFunc(double v) {
			switch (func) {
			case Linear:
				return 1;
			case Siglog:
				return v * (1 - v);
			case Hypertan:
				return 1 / pow(cosh(v), 2);
			}

			throw invalid_argument("There is no derivative for this activation function type.");
		}
	};
}