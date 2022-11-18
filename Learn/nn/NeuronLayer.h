#pragma once

#include <vector>
#include <random>

namespace nn {
	enum class ScalarFunc {
		Step, Linear, Siglog, Hypertan, ReLU, LeakyReLU, GeLU,
	};

	class INeuronLayer {
	protected:
		INeuronLayer() {}

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

		string layerName = "";

	public:
		INeuronLayer(int count, string name = "Layer") {
			if (count == 0) throw out_of_range("Invalid neuron count, cannot be zero.");

			neuronCount = count;
			layerName = name;
		}

		INeuronLayer(int count, bool independentInputs, bool useInputs, string name = "Layer")
			: INeuronLayer(count, name) {
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

		virtual double activationFunc(double v) = 0;
		virtual double derivActivationFunc(double v) = 0;
	};

	template<ScalarFunc Func>
	class FFNeuronLayer : public INeuronLayer {
		double activationFunc(double v) override = 0;
		double derivActivationFunc(double v) override = 0;
	};

	template<>
	class FFNeuronLayer<ScalarFunc::Step> : public INeuronLayer {
	public:
		FFNeuronLayer(int count, string name = "Layer")
			: INeuronLayer(count, name) {}
		FFNeuronLayer(int count, bool independentInputs, bool useInputs, string name = "Layer")
			: INeuronLayer(count, independentInputs, useInputs, name) {}

		double activationFunc(double v) override;
		double derivActivationFunc(double v) override;
	};

	template<>
	class FFNeuronLayer<ScalarFunc::Linear> : public INeuronLayer {
	public:
		FFNeuronLayer(int count, string name = "Layer")
			: INeuronLayer(count, name) {}
		FFNeuronLayer(int count, bool independentInputs, bool useInputs, string name = "Layer")
			: INeuronLayer(count, independentInputs, useInputs, name) {}

		double activationFunc(double v) override;
		double derivActivationFunc(double v) override;
	};

	template<>
	class FFNeuronLayer<ScalarFunc::Siglog> : public INeuronLayer {
	public:
		FFNeuronLayer(int count, string name = "Layer")
			: INeuronLayer(count, name) {}
		FFNeuronLayer(int count, bool independentInputs, bool useInputs, string name = "Layer")
			: INeuronLayer(count, independentInputs, useInputs, name) {}

		double activationFunc(double v) override;
		double derivActivationFunc(double v) override;
	};

	template<>
	class FFNeuronLayer<ScalarFunc::Hypertan> : public INeuronLayer {
	public:
		FFNeuronLayer(int count, string name = "Layer")
			: INeuronLayer(count, name) {}
		FFNeuronLayer(int count, bool independentInputs, bool useInputs, string name = "Layer")
			: INeuronLayer(count, independentInputs, useInputs, name) {}

		double activationFunc(double v) override;
		double derivActivationFunc(double v) override;
	};

	template<>
	class FFNeuronLayer<ScalarFunc::ReLU> : public INeuronLayer {
	public:
		FFNeuronLayer(int count, string name = "Layer")
			: INeuronLayer(count, name) {}
		FFNeuronLayer(int count, bool independentInputs, bool useInputs, string name = "Layer")
			: INeuronLayer(count, independentInputs, useInputs, name) {}

		double activationFunc(double v) override;
		double derivActivationFunc(double v) override;
	};

	template<>
	class FFNeuronLayer<ScalarFunc::LeakyReLU> : public INeuronLayer {
	public:
		FFNeuronLayer(int count, string name = "Layer")
			: INeuronLayer(count, name) {}
		FFNeuronLayer(int count, bool independentInputs, bool useInputs, string name = "Layer")
			: INeuronLayer(count, independentInputs, useInputs, name) {}

		double activationFunc(double v) override;
		double derivActivationFunc(double v) override;
	};

	template<>
	class FFNeuronLayer<ScalarFunc::GeLU> : public INeuronLayer {
	public:
		FFNeuronLayer(int count, string name = "Layer")
			: INeuronLayer(count, name) {}
		FFNeuronLayer(int count, bool independentInputs, bool useInputs, string name = "Layer")
			: INeuronLayer(count, independentInputs, useInputs, name) {}

		double activationFunc(double v) override;
		double derivActivationFunc(double v) override;
	};
}