#pragma once

namespace nn {
	enum TrainingType {
		Perceptron, Adaline
	};

	class SupervisedNetworkTrainer {
	private:
		int				epochTarget;
		double			errorTarget;
		double			learningRate;
		TrainingType	trainingType;

		double cost(int n, double* nnEstimate, double* actual) {
			double sum = 0;

			for (int i = 0; i < n; i++) {
				sum += pow(nnEstimate[i] - actual[i], 2);
			}
			
			return 1 / n * sum;
		}

		// 
		double deltaW(double target, double result, double input, double adalineDeriv) {

			// rate * (target - result) * input
			switch (trainingType) {
			case Perceptron:
				return learningRate * (target - result) * input;
			case Adaline:
				return learningRate * (target - result) * input * adalineDeriv;
			}
		}

	public:
		void train() {
			
		}
	};
};