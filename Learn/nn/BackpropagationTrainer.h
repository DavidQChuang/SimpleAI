#pragma once

#include "SupervisedNetworkTrainer.h"

namespace nn {
	class BackpropagationTrainer : public SupervisedNetworkTrainer {
	protected:
		bool checkTrainingInputs(NeuralNetwork& network,
			double* inputs, size_t inLength, double* expOutputs, size_t outLength) override {
			throw invalid_argument("unimplemented");
		}

		void trainOnEpoch(NeuralNetwork& network, double* inputs, double* expOutputs, double* buffer, double* outPtr) {
			throw invalid_argument("unimplemented");
		}

	public:
		BackpropagationTrainer(double learnRate = 0.1, double error = 0.002, int epochs = 1000)
			: SupervisedNetworkTrainer(learnRate, error, epochs) { }
	};
}