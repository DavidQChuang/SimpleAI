#pragma once

#include <vector>
#include <random>

namespace nn {
	enum class ActFunc {
		Step, Linear, Siglog, Hypertan, ReLU, LeakyReLU, GeLU,
	};

	template<ActFunc Func>
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

		bool overrideUseInputs = false;
		bool overrideUseOutputs = false;
		bool overrideIndependentInputs = false;

		string	layerName = "";

	public:
		NeuronLayer(int count, string name = "Layer") {
			if (count == 0) throw out_of_range("Invalid neuron count, cannot be zero.");

			neuronCount = count;
			layerName = name;
		}

		NeuronLayer(int count, bool independentInputs, bool useInputs, string name = "Layer")
			: NeuronLayer(count, name) {
			overrideUseInputs = true;
			overrideUseOutputs = true;
			overrideIndependentInputs = true;

			mUseInputs = useInputs;
			mUseOutputs = false;
			mIndependentInputs = independentInputs;
		}

		void init(int inputsPerNeuron, int outputsPerNeuron, bool independentInputs, bool useInputs, bool useOutputs) {
			if (inputsPerNeuron < 0) throw invalid_argument("Layer must have at least 0 inputs per neuron.");
			if (outputsPerNeuron < 0) throw invalid_argument("Layer must have at least 0 outputs per neuron.");

			mNeuronInputs = inputsPerNeuron;
			mNeuronOutputs = outputsPerNeuron;

			if (!overrideUseInputs) {
				mUseInputs = useInputs;
			}
			if (!overrideUseOutputs) {
				mUseOutputs = useOutputs;
			}
			if (!overrideIndependentInputs) {
				mIndependentInputs = independentInputs;
			}

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

		inline bool independentInputs() { return mIndependentInputs; }

		inline vector<double>& weightsIn() {
			if (mUseInputs) {
				return inputWeights;
			}
			else {
				int inputs = mNeuronInputs * neuronCount;
				auto weights = vector<double>(inputs);
				for (int i = 0; i < inputs; i++) {
					weights[i] = 1;
				}

				return weights;
			}
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
					printf("\n%-7d | IN : [ ", n);
					for (int i = 0; i < mNeuronInputs; i++) {
						printf("%11.8f", inputWeights[n * mNeuronInputs + i]);

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
						printf("%11.8f", outputWeights[n * mNeuronOutputs + i]);

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

		/*double vectorActivationFunc(double* output, ) {
			double res = 0;

			switch (func) {
			case ActFunc::Softmax:
				res = v < 0 ? 0 : 1;
				break;
			}

			return res;
		}*/

		double activationFunc(double v) = 0;
		double derivActivationFunc(double v) = 0;
	};

#define NAN_V_MSG "Activation function input was NaN."
#define NAN_DV_MSG "Activation function deriv input was NaN."
#define CHECK_NAN(v, msg) if(v != v) throw invalid_argument(msg)

	template<>
	class NeuronLayer<ActFunc::Step> {
	public:
		double activationFunc(double v) {
			CHECK_NAN(v, NAN_V_MSG);

			return !signbit(v);
		}

		double derivActivationFunc(double v) {
			CHECK_NAN(v, NAN_DV_MSG);

			return 0;
		}
	};

	template<>
	class NeuronLayer<ActFunc::Linear> {
	public:
		double activationFunc(double v) {
			CHECK_NAN(v, NAN_V_MSG);

			return v;
		}

		double derivActivationFunc(double v) {
			CHECK_NAN(v, NAN_DV_MSG);

			return 1;
		}
	};

	template<>
	class NeuronLayer<ActFunc::Siglog> {
	public:
		double activationFunc(double v) {
			CHECK_NAN(v, NAN_V_MSG);

			return 1.0 / (1.0 + exp(-v));
		}

		double derivActivationFunc(double v) {
			CHECK_NAN(v, NAN_DV_MSG);

			return v * (1.0 - v);
		}
	};

	template<>
	class NeuronLayer<ActFunc::Hypertan> {
	public:
		double activationFunc(double v) {
			CHECK_NAN(v, NAN_V_MSG);

			return tanh(v);
		}

		double derivActivationFunc(double v) {
			CHECK_NAN(v, NAN_DV_MSG);

			return 1.0 / pow(cosh(v), 2);
		}
	};

	template<>
	class NeuronLayer<ActFunc::ReLU> {
	public:
		double activationFunc(double v) {
			return max(0.0, v);
		}

		double derivActivationFunc(double v) {
			CHECK_NAN(v, NAN_DV_MSG);

			return !signbit(v);
		}
	};

	template<>
	class NeuronLayer<ActFunc::LeakyReLU> {
	public:
		double activationFunc(double v) {
			return max(0.01 * v, v);
		}

		double derivActivationFunc(double v) {
			CHECK_NAN(v, NAN_DV_MSG);

			return signbit(v) * 0.01 + !signbit(v);
		}
	};

#undef NAN_V_MSG
#undef NAN_DV_MSG
#undef CHECK_NAN
}