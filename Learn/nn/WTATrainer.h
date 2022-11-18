#pragma once

#include "UnsupervisedTrainer.h"

namespace nn {
	class WTATrainer : public UnsupervisedTrainer {
	private:

	protected:
		void initTrainingSet(NeuralNetwork& network, double* inputs, size_t inLength) override {
			UnsupervisedTrainer::initTrainingSet(network, inputs, inLength);

			if (network.depth() > 2)
				throw invalid_argument("Winner-takes-all trainer requires 1 inout layer or 1 in + 1 out layer. ");
		}

		void trainOnEpoch(NeuralNetwork& network, double* inputs, double* buffer, double* outPtr) {
			NeuralNetwork::Layer& outputLayer = network.getLayer(network.depth() - 1);
			vector<double>& weightsIn = outputLayer.weightsIn();

			int neuronCount = outputLayer.size();
			int inputCount = outputLayer.inputsPerNeuron();
			int outputCount = outputLayer.expectedOutputs();

			int nWinner = 0;
			double sumWinner = DBL_MIN;
			for (int n = 0; n < neuronCount; n++) {
				double sum = 0;
				for (int o = 0; o < outputCount; o++) {
					sum += outPtr[n * outputCount + o];
				}

				if (sum > sumWinner) {
					nWinner = n;
					sumWinner = sum;
				}
			}
			
			// update inputs of winner neuron
			for (int i = 0; i < inputCount; i++) {
				int w = nWinner * inputCount + i;

				weightsIn[w] += learningRate * (inputs[i] - weightsIn[w]);
			}
		}

	public:
		WTATrainer(double learnRate = 0.1, double error = 0.002, int epochs = 1000)
			: UnsupervisedTrainer(learnRate, error, epochs) { }
	};
}