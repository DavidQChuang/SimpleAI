#pragma once

#include <vector>
#include <random>

namespace nn {
	enum class ActFunc {
		Step, Linear, Siglog, Hypertan
	};

	class NeuronLayer {
	private:
		int		mNeuronInputs = 0; // input per neuron
		int		mNeuronOutputs = 0; // output per neuron
		bool	mUseInputs = true;
		bool	mUseOutputs = true;
		int		neuronCount = 0;

		vector<double> inputWeights;
		vector<double> outputWeights;

		ActFunc func = ActFunc::Step;

		string	layerName = "";

	public:
		NeuronLayer(int count, ActFunc f, string name = "Layer") {
			if (count == 0) throw out_of_range("Invalid neuron count, cannot be zero.");

			neuronCount = count;
			func = f;
			layerName = name;
		}

		void init(int inputsPerNeuron, int outputsPerNeuron, bool useInputs, bool useOutputs) {
			if (inputsPerNeuron < 1) throw invalid_argument("Uninitialized layer.");
			if (outputsPerNeuron < 1) throw invalid_argument("Uninitialized layer.");

			mNeuronInputs = inputsPerNeuron;
			mNeuronOutputs = outputsPerNeuron;

			mUseInputs = useInputs;
			mUseOutputs = useOutputs;

			initRandomWeights();
		}

		void initRandomWeights() {
			if (mUseInputs) {
				int inputs = expectedInputs();
				inputWeights = vector<double>(inputs);
				for (int i = 0; i < inputs; i++) {
					inputWeights[i] = (double)rand() / RAND_MAX;
				}
			}

			if (mUseOutputs) {
				int outputs = expectedOutputs();
				outputWeights = vector<double>(outputs);
				for (int i = 0; i < outputs; i++) {
					outputWeights[i] = (double)rand() / RAND_MAX;
				}
			}
		}

		void initWeights(double value) {
			if (mUseInputs) {
				int inputs = expectedInputs();
				inputWeights = vector<double>(inputs);
				for (int i = 0; i < inputs; i++) {
					inputWeights[i] = value;
				}
			}

			if (mUseOutputs) {
				int outputs = expectedOutputs();
				outputWeights = vector<double>(outputs);
				for (int i = 0; i < outputs; i++) {
					outputWeights[i] = value;
				}
			}
		}

		inline int expectedInputs() { return mNeuronInputs * neuronCount; }
		inline int expectedOutputs() { return mNeuronOutputs * neuronCount; }
		
		inline int neuronInputs() { return mNeuronInputs; }
		inline int neuronOutputs() { return mNeuronOutputs; }

		inline int size() { return neuronCount; }
		inline string name() { return layerName; }

		inline vector<double>& weightsIn() {
			return inputWeights;
		}
		inline vector<double>& weightsOut() {
			return outputWeights;
		}

		void execute(double* input, int inputLength, double* output, int outputLength) {
			if (mNeuronInputs == 0) throw invalid_argument("Uninitialized layer.");

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

				// copy outputs to buffer
				if (mUseOutputs) {
					for (int i = 0; i < mNeuronOutputs; i++) {
						output[i] = activationFunc(sum * outputWeights[outWOffset++]);
					}
				}
				else {
					for (int i = 0; i < mNeuronOutputs; i++) {
						output[i] = activationFunc(sum);
					}
				}

				input += mNeuronInputs;
				output += mNeuronOutputs;
			}
		}

		void display() {
			if (mNeuronInputs == 0) {
				printf("Uninitialized layer: %dx%d inputs, %dx%d outputs",
					mNeuronInputs, neuronCount, mNeuronOutputs, neuronCount);
				return;
			}

			printf("\nLayer [%dx(%d,%d)]", neuronCount, mNeuronInputs, mNeuronOutputs);
			printf("\n%-7s | -", "Neurons");

			for (int n = 0; n < neuronCount; n++) {
				if (mUseInputs) {
					printf("\n%-7d | IN: [ ", n);
					for (int i = 0; i < mNeuronInputs; i++) {
						printf("%.8f", inputWeights[n * mNeuronInputs + i]);

						if (i + 1 != mNeuronInputs) {
							printf(", ");
						}
						else break;
					}
					printf(" ]");
				}

				if (mUseOutputs) {
					printf("\n%-7d | OUT: [ ", n);
					for (int i = 0; i < mNeuronOutputs; i++) {
						printf("%.8f", outputWeights[n * mNeuronOutputs + i]);

						if (i + 1 != mNeuronOutputs) {
							printf(", ");
						}
						else break;
					}
					printf(" ]");
				}
			}
			printf("\n");
		}

		double activationFunc(double v) {
			switch (func) {
			case ActFunc::Step:
				return v < 0 ? 0 : 1;
				break;
			case ActFunc::Linear:
				return v;
			case ActFunc::Siglog:
				return 1 / (1 + exp(-v));
			case ActFunc::Hypertan:
				return (1 - exp(-v)) / (1 + exp(-v));
			}

			throw invalid_argument("There is no function for this activation function type.");
		}

		double derivActivationFunc(double v) {
			switch (func) {
			case ActFunc::Linear:
				return 1;
			case ActFunc::Siglog:
				return v * (1 - v);
			case ActFunc::Hypertan:
				return 1 / pow(cosh(v), 2);
			}

			throw invalid_argument("There is no derivative for this activation function type.");
		}
	};
}