#pragma once

#include <cmath>
#include "../NeuralNetwork.h"

namespace nn {
	template<typename... LayerArgs>
	class UnsupervisedTrainer {
	protected:
		static_assert((std::is_base_of<INeuronLayer, LayerArgs>::value && ...),
			"Arguments must be derived from INeuronLayer.");

		int				epochTarget;
		double			errorTarget;
		double			learningRate;

		double cost(int n, double* nnEstimate, double* actual) {
			double sum = 0;

			for (int i = 0; i < n; i++) {
				sum += pow(nnEstimate[i] - actual[i], 2);
			}

			return sum / n;
		}

		// 
		//double deltaW(double target, double result, double input, double adalineDeriv) {
		//	// rate * (target - result) * input
		//	switch (trainingType) {
		//	case Perceptron:
		//		return learningRate * (target - result) * input;
		//	case Adaline:
		//		return learningRate * (target - result) * input * adalineDeriv;
		//	}
		//}

	protected:

		virtual void trainOnEpoch(FFNeuralNetwork<LayerArgs...>& network, double* inputs, double* buffer, double* outPtr) = 0;

		virtual void initTrainingSet(FFNeuralNetwork<LayerArgs...>& network, double* inputs, size_t inLength) {
			if (network.expectedInputs() != inLength)
				throw invalid_argument("Input of network and size of input buffer don't match.");
		}

		virtual void cleanUp() {}

	public:
		UnsupervisedTrainer(double learnRate = 0.1, double error = 0.002, int epochs = 1000) {
			learningRate = learnRate;
			errorTarget = error;
			epochTarget = epochs;
		}

		void setLearningRate(double rate) {
			learningRate = rate;
		}

		void setErrorTarget(double error) {
			errorTarget = error;
		}

		void setEpochTarget(int epochs) {
			epochTarget = epochs;
		}

		void train(FFNeuralNetwork<LayerArgs...>& network, int trainingSets, double** inputSet, size_t inLength) {

			int bufferSize = network.expectedBufferSize();

			unique_ptr<double[]> bufferPtr(new double[bufferSize]);
			double* buffer = bufferPtr.get();

			int e = 0;
			try {
				while (e < epochTarget) {
					for (int i = 0; i < trainingSets; i++) {
						double* inputs = inputSet[i];
						double* outPtr = executeOnSet(network, buffer,
							inputs, inLength);

						trainOnEpoch(network, inputs, buffer, outPtr);
					}

					e++;
				}
			}
			catch (exception ex) {
				printf("\n\n!!! ERROR: Threw exception while training: %s", ex.what());
				printf("\nFailed on epoch %d", e);
				displayResults(network, buffer, trainingSets,
					inputSet, inLength, e);
				return;
			}

			cleanUp();

			displayResults(network, buffer, trainingSets,
				inputSet, inLength, e);
		}

		void train(FFNeuralNetwork<LayerArgs...>& network, double* inputs, size_t inLength) {
			unique_ptr<double*[]> inputSet(new double*[1] { inputs });

			train(network, 1, inputSet.get(), inLength);
		}

	private:
		double* executeOnSet(FFNeuralNetwork<LayerArgs...>& network, double* buffer,
			double* inputs, size_t inLength) {

			initTrainingSet(network, inputs, inLength);
			memcpy(buffer, inputs, inLength * sizeof(double));
			return network.executeToIOArray(buffer, inLength, network.expectedBufferSize());
		}

		void displayResults(FFNeuralNetwork<LayerArgs...>& network, double* buffer,
			int trainingSets, double** inputSet, int inLength,
			int e) {
			bool failed = false;

			for (int i = 0; i < trainingSets; i++) {
				double* inputs = inputSet[i];

				printf("\n\n### Training set #%d\n", i);
				printf("\n%-10s | [ ", "Inputs");
				for (int i = 0;;) {
					printf("%.3f", inputs[i]);

					if (++i < inLength) {
						printf(", ");
					}
					else break;
				}
				printf(" ]");

				try {
					double* outPtr = executeOnSet(network, buffer,
						inputs, inLength);

					printf("\n%-10s | [ ", "NNOutputs");
					for (int i = 0;;) {
						printf("%.6f", outPtr[i]);

						if (++i < network.expectedOutputs()) {
							printf(", ");
						}
						else break;
					}
					printf(" ]");
				}
				catch (exception ex) {
					printf("\n%-10s | [ FAILED ]", "NNOutputs");
					failed = true;
				}
			}
			printf("\n\n### Final Results");

			if (failed) {
				printf("%-10s | %-30s | Epoch %-3d", "Result", "[ FAILED ]", e);
			}
			else if (e == epochTarget) {
				printf("%-10s | %-30s | Epoch %-3d", "Result", "Reached epoch limit", e);
			}
			else {
				printf("%-10s | %-30s | Epoch %-3d", "Result", "Reached minimum rate target", e);
			}
			printf("\n");
		}
	};
}