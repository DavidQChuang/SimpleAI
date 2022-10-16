#pragma once

#include <vector>

using namespace std;

namespace nn {
	enum ActivationFunction {
		Step, Linear, Siglog, Hypertan
	};

	class Neuron {
	private:
		ActivationFunction func;

		vector<double> weightIn;
		vector<double> weightOut;

	public:
		Neuron(ActivationFunction func, vector<double> weightIn, vector<double> weightOut)
			: func(func), weightIn(weightIn), weightOut(weightOut) {}

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

		vector<double> weightsIn() { return weightIn; }
		vector<double> weightsOut() { return weightOut; }

		void display() {
			vector<double>::iterator it;

			printf("Input weights:\n[");
			for (it = weightIn.begin(); it < weightIn.end(); it++) {
				double v = *it;
				printf("%s, ", to_string(v).c_str());
			}
			printf("]\n");

			printf("Output weights:\n");
			for (it = weightOut.begin(); it < weightOut.end(); it++) {
				double v = *it;
				printf("%s, ", to_string(v).c_str());
			}
			printf("]\n");

		}
	};
}