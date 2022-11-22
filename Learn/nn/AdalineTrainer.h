#pragma once

#include "SupervisedTrainer.h"

namespace nn {
	class AdalineTrainer : public SupervisedTrainer {
	private:
		double momentum;
		vector<double> prevWeightDeltas;

		INeuronLayer* layer;
		vector<double>* weightsInPtr;
		int neurons;
		int inputOffset;

	protected:
		void initTraining(NeuralNetwork& network,
			int trainingSets,
			double** inputSet, size_t inLength,
			double** expOutputSet, size_t outLength)
		override {
			SupervisedTrainer::initTraining(network, trainingSets,
				inputSet, inLength, expOutputSet, outLength);

			if (network.depth() > 2)
				throw invalid_argument("Adaline requires 1 inout layer or 1 in + 1 out layer. ");

			if (outLength != 1)
				throw invalid_argument("Adaline requires 1 output.");

			layer = &network.getLayer(0);
			weightsInPtr = &layer->weightsIn();
			neurons = layer->size();

			inputOffset = network.depth() == 1 ? 0 : network.expectedInputs();

			for (int n = 0; n < neurons; n++) {
				for (int i = 0; i < layer->inputsPerNeuron(); i++) {
					prevWeightDeltas.push_back(0);
				}
			}
		}

		void trainOnSet(NeuralNetwork& network,
			double* inputs, double* expOutputs,
			double* buffer, double* outPtr)
		override {
			double error = expOutputs[0] - outPtr[0]; // target - result, positive if result was lower, negative if result was higher

			int inputCount = layer->inputsPerNeuron();
			int outputCount = layer->outputsPerNeuron();

			vector<double>& weightsIn = *weightsInPtr;
			double* inPtr = buffer + inputOffset;

			int in = 0;
			double sum = 0;
			for (int n = 0; n < neurons; n++) {
				for (int i = 0; i < inputCount; i++) {
					int w = n * inputCount + i;

					sum += inPtr[in] * weightsIn[w];
					in++;
				}
				if (!layer->independentInputs()) {
					in -= inputCount;
				}
			}

			in = 0;
			int wd = 0;
			for (int n = 0; n < neurons; n++) {
				for (int i = 0; i < inputCount; i++) {
					int w = n * inputCount + i;

					double weightDelta = learningRate * error * inPtr[in] * layer->derivActivationFunc(sum)
						+ momentum * prevWeightDeltas[wd];

					weightsIn[w] += weightDelta;

					prevWeightDeltas[wd] = weightDelta;

					in++;
					wd++;
				}
				if (!layer->independentInputs()) {
					in -= inputCount;
				}
			}
		}

	public:
		AdalineTrainer(double learnRate = 0.1, double error = 0.002, int epochs = 1000, double momentum = 0.5)
			: SupervisedTrainer(learnRate, error, epochs), momentum(momentum) { }
	};
}