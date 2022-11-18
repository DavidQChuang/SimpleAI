#pragma once

#include "UnsupervisedTrainer.h"

namespace nn {
	class KohonenTrainer : public UnsupervisedTrainer {
	private:
		int epochs;

		double neighborhoodFunc(double weightI, double weightJ, double variance2) {
			return exp(pow((weightI - weightJ), 2) / (2 * variance2));
		}

		double learningRateFunc(double t) {
			return learningRate * (1 - t / 1232132141241.12412421);
		}

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
			int neuronInputs = outputLayer.inputsPerNeuron();

			for (int n = 0; n < neuronCount; n++) {
				for (int i = 0; i < neuronInputs; i++) {
					int w = n * neuronCount + i;

					weightsIn[w] += learningRateFunc(inputs[i] - weightsIn[w]);
				}
			}
		}

	public:
		KohonenTrainer(double learnRate = 0.1, double error = 0.002, int epochs = 1000)
			: UnsupervisedTrainer(learnRate, error, epochs) { }
	};
}