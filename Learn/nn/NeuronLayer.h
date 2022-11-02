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
		bool	mIndependentInputs = false;
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

		void init(int inputsPerNeuron, int outputsPerNeuron, bool independentInputs, bool useInputs, bool useOutputs) {
			if (inputsPerNeuron < 1) throw invalid_argument("Uninitialized layer.");
			if (outputsPerNeuron < 1) throw invalid_argument("Uninitialized layer.");

			mNeuronInputs = inputsPerNeuron;
			mNeuronOutputs = outputsPerNeuron;

			mIndependentInputs = independentInputs;
			mUseInputs = useInputs;
			mUseOutputs = useOutputs;

			initRandomWeights();
		}

		void initRandomWeights() {
			if (mUseInputs) {
				int inputs = mNeuronInputs * neuronCount;
				inputWeights = vector<double>(inputs);
				for (int i = 0; i < inputs; i++) {
					inputWeights[i] = (double)rand() / RAND_MAX;
				}
			}

			if (mUseOutputs) {
				int outputs = mNeuronOutputs * neuronCount;
				outputWeights = vector<double>(outputs);
				for (int i = 0; i < outputs; i++) {
					outputWeights[i] = (double)rand() / RAND_MAX;
				}
			}
		}

		void initWeights(double value) {
			if (mUseInputs) {
				int inputs = mNeuronInputs * neuronCount;
				inputWeights = vector<double>(inputs);
				for (int i = 0; i < inputs; i++) {
					inputWeights[i] = value;
				}
			}

			if (mUseOutputs) {
				int outputs = mNeuronOutputs * neuronCount;
				outputWeights = vector<double>(outputs);
				for (int i = 0; i < outputs; i++) {
					outputWeights[i] = value;
				}
			}
		}

		inline int expectedInputs() { // if inputs are independent, they don't overlap
			return mNeuronInputs * (neuronCount * mIndependentInputs + 1 - mIndependentInputs);
		}
		inline int expectedOutputs() { return mNeuronOutputs * neuronCount; }
		
		inline int inputsPerNeuron() { return mNeuronInputs; }
		inline int outputsPerNeuron() { return mNeuronOutputs; }

		inline int size() { return neuronCount; }
		inline string name() { return layerName; }

		inline bool independentInputs() { return mIndependentInputs;}

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

			int in = 0;
			int out = 0;
			for (int n = 0; n < neuronCount; n++) {
				double sum = 0;

				// sum weights * input
				if (mUseInputs) {
					for (int i = 0; i < mNeuronInputs; i++) {
						sum += input[in] * inputWeights[n * mNeuronInputs + i];
						in++;
					}
				}
				else {
					for (int i = 0; i < mNeuronInputs; i++) {
						sum += input[in];
						in++;
					}
				}

				if (!mIndependentInputs) { // if inputs are non-independent, all neurons ue the same inputs.
					in = 0;
				}

				// copy outputs to buffer
				if (mUseOutputs) {
					for (int i = 0; i < mNeuronOutputs; i++) {
						output[out] = activationFunc(sum * outputWeights[out]);
						out++;
					}
				}
				else {
					for (int i = 0; i < mNeuronOutputs; i++) {
						output[out] = activationFunc(sum);
						out++;
					}
				}
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
			double res;

			if (v != v)
				throw invalid_argument("Activation function input was NaN.");

			switch (func) {
			case ActFunc::Step:
				res =  v < 0 ? 0 : 1;
				break;

			case ActFunc::Linear:
				res =  v;
				break;

			case ActFunc::Siglog:
				res =  1 / (1 + exp(-v));
				break;

			case ActFunc::Hypertan:
				res =  (1 - exp(-v)) / (1 + exp(-v));
				break;

			default:
				throw invalid_argument("There is no function for this activation function type.");
			}

			if (res != res)
				throw invalid_argument("Activation function resulted in NaN.");
		}

		double derivActivationFunc(double v) {
			double res;

			if (v != v)
				throw invalid_argument("Activation function deriv input was NaN.");

			switch (func) {
			case ActFunc::Linear:
				res =  1;
				break;

			case ActFunc::Siglog:
				res =  v * (1 - v);
				break;

			case ActFunc::Hypertan:
				res = 1 / pow(cosh(v), 2);
				break;

			default:
				throw invalid_argument("There is no derivative for this activation function type.");
			}

			if (res != res)
				throw invalid_argument("Activation function deriv resulted in NaN.");
		}
	};
}