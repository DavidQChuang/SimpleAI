#pragma once

#include <vector>
#include <random>
#include <string>
#include <stdexcept>

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

		std::vector<double> inputWeights;
		std::vector<double> outputWeights;

		bool overrideUseInputs = false;
		bool overrideUseOutputs = false;
		bool overrideIndependentInputs = false;

		std::string layerName = "";

	public:
		INeuronLayer(int count, std::string name = "Layer") {
			if (count == 0) throw std::out_of_range("Invalid neuron count, cannot be zero.");

			neuronCount = count;
			layerName = name;
		}

		INeuronLayer(int count, bool independentInputs, bool useInputs, std::string name = "Layer")
			: INeuronLayer(count, name) {
			overrideUseInputs = true;
			overrideUseOutputs = true;
			overrideIndependentInputs = true;

			mUseInputs = useInputs;
			mUseOutputs = false;
			mIndependentInputs = independentInputs;
		}

		virtual INeuronLayer* clone() = 0;

		void init(int inputsPerNeuron, int outputsPerNeuron, bool independentInputs, bool useInputs, bool useOutputs) {
			if (inputsPerNeuron < 0) throw std::invalid_argument("Layer must have at least 0 inputs per neuron.");
			if (outputsPerNeuron < 0) throw std::invalid_argument("Layer must have at least 0 outputs per neuron.");

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
				inputWeights = std::vector<double>(inputs);
				for (int i = 0; i < inputs; i++) {
					inputWeights[i] = (double)rand() / RAND_MAX;
				}
			}

			if (mUseOutputs) {
				int outputs = mNeuronOutputs * neuronCount;
				outputWeights = std::vector<double>(outputs);
				for (int i = 0; i < outputs; i++) {
					outputWeights[i] = (double)rand() / RAND_MAX;
				}
			}
		}

		void initWeights(double value) {
			if (mUseInputs) {
				int inputs = mNeuronInputs * neuronCount;
				inputWeights = std::vector<double>(inputs);
				for (int i = 0; i < inputs; i++) {
					inputWeights[i] = value;
				}
			}

			if (mUseOutputs) {
				int outputs = mNeuronOutputs * neuronCount;
				outputWeights = std::vector<double>(outputs);
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
		inline std::string name() { return layerName; }

		inline bool independentInputs() { return mIndependentInputs; }

		inline std::vector<double>& weightsIn() {
			if (mUseInputs) {
				return inputWeights;
			}
			else {
				int inputs = mNeuronInputs * neuronCount;
				auto weights = std::vector<double>(inputs);
				for (int i = 0; i < inputs; i++) {
					weights[i] = 1;
				}

				return weights;
			}
		}

		inline std::vector<double>& weightsOut() {
			return outputWeights;
		}

		void execute(double* input, int inputLength, double* output, int outputLength) {
			if (mNeuronInputs == 0) throw std::invalid_argument("Uninitialized layer.");

			if (input == NULL) throw std::invalid_argument("Null input pointer.");
			if (output == NULL) throw std::invalid_argument("Null output pointer.");

			if (inputLength != expectedInputs()) throw std::invalid_argument("Input buffer length is invalid.");
			if (outputLength != expectedOutputs()) throw std::invalid_argument("Output buffer length is invalid.");

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

#define DEFINE_LAYER(className) \
	template<>\
	class className : public INeuronLayer {\
	public:\
		className(int count, std::string name = "Layer")\
			: INeuronLayer(count, name) {}\
		className(int count, bool independentInputs, bool useInputs, std::string name = "Layer")\
			: INeuronLayer(count, independentInputs, useInputs, name) {}\
\
		double activationFunc(double v) override;\
		double derivActivationFunc(double v) override;\
\
		INeuronLayer* clone() override { return new className(*this); }\
	}

	DEFINE_LAYER(FFNeuronLayer<ScalarFunc::Step>);
	DEFINE_LAYER(FFNeuronLayer<ScalarFunc::Linear>);
	DEFINE_LAYER(FFNeuronLayer<ScalarFunc::Siglog>);
	DEFINE_LAYER(FFNeuronLayer<ScalarFunc::Hypertan>);
	DEFINE_LAYER(FFNeuronLayer<ScalarFunc::ReLU>);
	DEFINE_LAYER(FFNeuronLayer<ScalarFunc::LeakyReLU>);
	DEFINE_LAYER(FFNeuronLayer<ScalarFunc::GeLU>);
}