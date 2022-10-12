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

		double activationFunc(double x) {
			switch (func) {
			case Step:
				return x < 0 ? 0 : 1;
				break;
			case Linear:
				return x;
			case Siglog:
				return 1 / (1 + exp(-x));
			case Hypertan:
				return (1 - exp(-x)) / (1 + exp(-x));
			}
		}

		double derivActivationFunc(double x) {
			
		}

		vector<double> weightIn() { return weightIn; }
		vector<double> weightOut() { return weightOut; }

		void display() {
			vector<double>::iterator it;

			printf("Input weights:\n[");
			for (it = weightIn.begin(); it < weightIn.end(); it++) {
				double v = *it;
				printf("%s, ", to_string(v));
			}
			printf("]\n");

			printf("Output weights:\n");
			for (it = weightOut.begin(); it < weightOut.end(); it++) {
				double v = *it;
				printf("%s, ", to_string(v));
			}
			printf("]\n");

		}
	};
}