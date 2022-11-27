#pragma once

#include "UnsupervisedTrainer.h"

namespace nn {
	template<typename... LayerArgs>
	class KohonenTrainer : public UnsupervisedTrainer<LayerArgs...> {
	private:
		vector<double> distances;

		double neighborhoodFunc(double weightI, double weightJ, double variance) {
			return exp(pow((weightI - weightJ), 2) / (2 * variance * 1));
		}

		double learningRateFunc(double t) {
			throw invalid_argument("uninpmlemented");
			return this->learningRate * (1 - t / 1);
		}

	protected:
		void initTrainingSet(FFNeuralNetwork<LayerArgs...>& network, double* inputs, size_t inLength) override {
			UnsupervisedTrainer<LayerArgs...>::initTrainingSet(network, inputs, inLength);

			if (network.depth() > 2)
				throw invalid_argument("Winner-takes-all trainer requires 1 inout layer or 1 in + 1 out layer. ");
		}

		void trainOnEpoch(FFNeuralNetwork<LayerArgs...>& network, double* inputs, double* buffer, double* outPtr) override {
			NeuralNetwork::Layer& outputLayer = network.getLayer(network.depth() - 1);
			vector<double>& weightsIn = outputLayer.weightsIn();

			int neuronCount = outputLayer.size();
			int neuronInputs = outputLayer.inputsPerNeuron();

			throw invalid_argument("uninpmlemented");

			int winnerNeuron = 0;
			for (int n = 0; n < neuronCount; n++) {
				for (int i = 0; i < neuronInputs; i++) {
					int w = n * neuronCount + i;
				}
			}

			for (int n = winnerNeuron; n < neuronCount; n++) {
				for (int i = 0; i < neuronInputs; i++) {
					int w = winnerNeuron * neuronCount + i;

					double Wij = weightsIn[w];
					//weightsIn[w] += neighborhoodFunc(Wij, );
				}
			}
		}

	public:
		KohonenTrainer(double learnRate = 0.1, double error = 0.002, int epochs = 1000)
			: UnsupervisedTrainer<LayerArgs...>(learnRate, error, epochs) {}
	};
}