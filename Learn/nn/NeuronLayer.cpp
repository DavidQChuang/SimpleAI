#include "NeuronLayer.h"

#define _USE_MATH_DEFINES
#include <cmath>

#define CHECK_NAN(v, msg) if(v != v) throw std::invalid_argument(msg)

namespace nn {
	using namespace std;

	constexpr const char* NAN_V_MSG = "Activation function input was NaN.";
	constexpr const char* NAN_DV_MSG = "Activation function deriv input was NaN.";

	double FFNeuronLayer<ScalarFunc::Step>::activationFunc(double v) {
		CHECK_NAN(v, NAN_V_MSG);

		return !signbit(v);
	}

	double FFNeuronLayer<ScalarFunc::Step>::derivActivationFunc(double v) {
		CHECK_NAN(v, NAN_DV_MSG);

		return 0;
	}

	double FFNeuronLayer<ScalarFunc::Linear>::activationFunc(double v) {
		CHECK_NAN(v, NAN_V_MSG);

		return v;
	}

	double FFNeuronLayer<ScalarFunc::Linear>::derivActivationFunc(double v) {
		CHECK_NAN(v, NAN_DV_MSG);

		return 1;
	}


	double FFNeuronLayer<ScalarFunc::Siglog>::activationFunc(double v) {
		CHECK_NAN(v, NAN_V_MSG);

		return 1.0 / (1.0 + exp(-v));
	}

	double FFNeuronLayer<ScalarFunc::Siglog>::derivActivationFunc(double v) {
		CHECK_NAN(v, NAN_DV_MSG);

		return v * (1.0 - v);
	}


	double FFNeuronLayer<ScalarFunc::Hypertan>::activationFunc(double v) {
		CHECK_NAN(v, NAN_V_MSG);

		return tanh(v);
	}

	double FFNeuronLayer<ScalarFunc::Hypertan>::derivActivationFunc(double v) {
		CHECK_NAN(v, NAN_DV_MSG);

		return 1.0 / pow(cosh(v), 2);
	}


	double FFNeuronLayer<ScalarFunc::ReLU>::activationFunc(double v) {
		return max(0.0, v);
	}

	double FFNeuronLayer<ScalarFunc::ReLU>::derivActivationFunc(double v) {
		CHECK_NAN(v, NAN_DV_MSG);

		return !signbit(v);
	}


	double FFNeuronLayer<ScalarFunc::LeakyReLU>::activationFunc(double v) {
		return max(0.01 * v, v);
	}

	double FFNeuronLayer<ScalarFunc::LeakyReLU>::derivActivationFunc(double v) {
		CHECK_NAN(v, NAN_DV_MSG);

		return signbit(v) * 0.01 + !signbit(v);
	}


	double FFNeuronLayer<ScalarFunc::GeLU>::activationFunc(double v) {
		double cdf = (1 + erf(v / sqrt(2))) / 2;

		return v * cdf;
	}

	double FFNeuronLayer<ScalarFunc::GeLU>::derivActivationFunc(double v) {
		CHECK_NAN(v, NAN_DV_MSG);

		static const double inv_sqrt_2pi = 0.3989422804014327;

		double cdf = (1 + erf(v / sqrt(2))) / 2;
		double pdf = inv_sqrt_2pi * std::exp(-0.5f * v * v);

		return cdf + v * pdf;
	}
}

#undef CHECK_NAN