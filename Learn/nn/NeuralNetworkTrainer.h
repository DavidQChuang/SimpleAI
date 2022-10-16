#pragma once

namespace nn {
	enum TrainingType {
		Perceptron, Adaline
	};

	class SupervisedNetworkTrainer {
	private:
		int epochs;
		double minError;

		double cost(int n, double* nnEstimate, double* actual) {
			double sum = 0;

			for (int i = 0; i < n; i++) {
				sum += pow(nnEstimate[i] - actual[i], 2);
			}
			
			return 1 / n * sum;
		}

	public:

	};
};