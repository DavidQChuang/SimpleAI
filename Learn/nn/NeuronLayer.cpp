#include "NeuronLayer.h"

#ifdef DISABLE_CHECKS
#define CHECK_NAN(v, msg)
#else
#define CHECK_NAN(v, msg) if(v != v) throw std::invalid_argument(msg)
#endif


namespace nn {
	using namespace std;

	constexpr const char* NAN_V_MSG = "Activation function input was NaN.";
	constexpr const char* NAN_DV_MSG = "Activation function deriv input was NaN.";

	////////////////////////
	// INEURONLAYER

	void INeuronLayer::init(int inputsPerNeuron, int outputsPerNeuron, bool independentInputs, bool useInputs) {
		if (inputsPerNeuron < 0) throw std::invalid_argument("Layer must have at least 0 inputs per neuron.");
		if (outputsPerNeuron < 0) throw std::invalid_argument("Layer must have at least 0 outputs per neuron.");

		mNeuronInputs = inputsPerNeuron;
		mNeuronOutputs = outputsPerNeuron;

		if (!overrideUseInputs) {
			mUseInputs = useInputs;
		}
		if (!overrideIndependentInputs) {
			mIndependentInputs = independentInputs;
		}

		initWeights<WeightInit::Uniform, double, double, int>(0, 1, 0);
	}

	void INeuronLayer::execute(double* input, int inputLength, double* output, int outputLength) {
		if (mNeuronInputs == 0) throw std::invalid_argument("Uninitialized layer.");

		if (input == NULL) throw std::invalid_argument("Null input pointer.");
		if (output == NULL) throw std::invalid_argument("Null output pointer.");

		if (inputLength != totalInputs()) throw std::invalid_argument("Input buffer length is invalid.");
		if (outputLength != totalOutputs()) throw std::invalid_argument("Output buffer length is invalid.");

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

			for (int i = 0; i < mNeuronOutputs; i++) {
				output[out] = activationFunc(sum, i);
				out++;
			}

			vectorActivationFunc(output, outputLength);
		}
	}

	void INeuronLayer::display() {
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
		}
		printf("\n");
	}

	////////////////////////
	// INEURONLAYER WEIGHT INIT FUNCTIONS

	template<>
	void INeuronLayer::initWeights<WeightInit::Constant, double>(double weight) {
		if (!mUseInputs && weight != 1.0) {
			initWeights<WeightInit::Constant, double>(1.0);
			return;
		}
		
		int inputs = mNeuronInputs * neuronCount;
		inputWeights = std::vector<double>(inputs);

		for (int i = 0; i < inputs; i++) {
			inputWeights[i] = weight;
		}
	}

	template<>
	void INeuronLayer::initWeights<WeightInit::Normal, double, double, int>(double stdev, double mean, int seed) {
		if (!mUseInputs) {
			initWeights<WeightInit::Constant, double>(1.0);
			return;
		}

		// -3.5 stdev - ~99.95% of points above
		const double xMin = statmath::probit(0.0005) * stdev;
		// 3.5 stdev - ~99.95% of points below
		const double xMax = statmath::probit(0.9995) * stdev;

		std::minstd_rand eng(seed);
		std::normal_distribution<double> dist(0, stdev);

		int inputs = mNeuronInputs * neuronCount;
		inputWeights = std::vector<double>(inputs);

		for (int i = 0; i < inputs; i++) {
			inputWeights[i] = min(max(dist(eng), xMin), xMax);
		}
	}

	template<>
	void INeuronLayer::initWeights<WeightInit::Uniform, double, double, int>(double min, double max, int seed) {
		if (!mUseInputs) {
			initWeights<WeightInit::Constant, double>(1.0);
			return;
		}

		std::minstd_rand eng(seed);
		std::uniform_real_distribution<double> dist(min, max);

		int inputs = mNeuronInputs * neuronCount;
		inputWeights = std::vector<double>(inputs);

		for (int i = 0; i < inputs; i++) {
			inputWeights[i] = dist(eng);
		}
	}

	////////////////////////
	// ACTIVATION FUNCTIONS

	double FFNeuronLayer<ScalarFunc::Step>::activationFunc(double v, int n) {
		CHECK_NAN(v, NAN_V_MSG);

		return !signbit(v);
	}

	double FFNeuronLayer<ScalarFunc::Step>::derivActivationFunc(double v, int n) {
		CHECK_NAN(v, NAN_DV_MSG);

		return 0;
	}

	double FFNeuronLayer<ScalarFunc::Linear>::activationFunc(double v, int n) {
		CHECK_NAN(v, NAN_V_MSG);

		return v;
	}

	double FFNeuronLayer<ScalarFunc::Linear>::derivActivationFunc(double v, int n) {
		CHECK_NAN(v, NAN_DV_MSG);

		return 1;
	}


	double FFNeuronLayer<ScalarFunc::Siglog>::activationFunc(double v, int n) {
		CHECK_NAN(v, NAN_V_MSG);

		return 1.0 / (1.0 + exp(-v));
	}

	double FFNeuronLayer<ScalarFunc::Siglog>::derivActivationFunc(double v, int n) {
		CHECK_NAN(v, NAN_DV_MSG);

		double x = 1.0 / (1.0 + exp(-v));

		return x * (1.0 - x);
	}


	double FFNeuronLayer<ScalarFunc::Hypertan>::activationFunc(double v, int n) {
		CHECK_NAN(v, NAN_V_MSG);

		return tanh(v);
	}

	double FFNeuronLayer<ScalarFunc::Hypertan>::derivActivationFunc(double v, int n) {
		CHECK_NAN(v, NAN_DV_MSG);

		return 1.0 / pow(cosh(v), 2);
	}


	double FFNeuronLayer<ScalarFunc::ReLU>::activationFunc(double v, int n) {
		return max(0.0, v);
	}

	double FFNeuronLayer<ScalarFunc::ReLU>::derivActivationFunc(double v, int n) {
		CHECK_NAN(v, NAN_DV_MSG);

		return !signbit(v);
	}


	double FFNeuronLayer<ScalarFunc::LeakyReLU>::activationFunc(double v, int n) {
		return max(0.01 * v, v);
	}

	double FFNeuronLayer<ScalarFunc::LeakyReLU>::derivActivationFunc(double v, int n) {
		CHECK_NAN(v, NAN_DV_MSG);

		return signbit(v) * 0.01 + !signbit(v);
	}


	double FFNeuronLayer<ScalarFunc::GeLU>::activationFunc(double v, int n) {
		double cdf = (1 + erf(v / sqrt(2))) / 2;

		return v * cdf;
	}

	double FFNeuronLayer<ScalarFunc::GeLU>::derivActivationFunc(double v, int n) {
		CHECK_NAN(v, NAN_DV_MSG);

		static const double inv_sqrt_2pi = 0.3989422804014327;

		double cdf = (1 + erf(v / sqrt(2))) / 2;
		double pdf = inv_sqrt_2pi * std::exp(-0.5f * v * v);

		return cdf + v * pdf;
	}

	double FFVNeuronLayer<VectorFunc::Softmax>::activationFunc(double v, int n) {
		CHECK_NAN(v, NAN_V_MSG);
		return v;
	}

	double FFVNeuronLayer<VectorFunc::Softmax>::derivActivationFunc(double v, int n) {
		CHECK_NAN(v, NAN_DV_MSG);

		double ex = weightedSumExps[n];
		double c = totalSum - ex;
		double num = c * ex;
		double den = c + ex;

		return num / (den * den);
	}

	void FFVNeuronLayer<VectorFunc::Softmax>::vectorActivationFunc(double* outputs, int outputLength) {
		double sum = 0;

		weightedSumExps.resize(outputLength);

		for (int i = 0; i < outputLength; i++) {
			double val = exp(outputs[i]);
			CHECK_NAN(val, NAN_V_MSG);

			sum += val;
			weightedSumExps[i] = val;
		}
		totalSum = sum;

		for (int i = 0; i < outputLength; i++) {
			outputs[i] = exp(outputs[i]) / sum;
		}
	}

	void FFVNeuronLayer<VectorFunc::Argmax>::vectorActivationFunc(double* outputs, int outputLength) {
		int maxIdx = 0;
		double max = outputs[0];

		for (int i = 0; i < outputLength; i++) {
			double output = outputs[i];
			CHECK_NAN(output, NAN_V_MSG);

			if (output > max) {
				maxIdx = i;
				max = output;
			}
			outputs[i] = 0;
		}

		outputs[maxIdx] = 1;
	}
}

#undef CHECK_NAN