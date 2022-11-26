#pragma once
#include <cmath>
#include <Eigen/Dense>
#include "../NeuralNetwork.h"

namespace nn {
	class SupervisedTrainer {
	protected:
		// exit conditions
		int				epochTarget;
		double			errorTarget;
		double			mseMax = 1;

		// learning rate
		double			learningRate;

		// fields set & used during training
		Eigen::VectorXd setError;
		int				currSet;

		double cost(int n, double* nnEstimate, double* actual) {
			double sum = 0;

			for (int i = 0; i < n; i++) {
				sum += pow(nnEstimate[i] - actual[i], 2);
			}

			return sum / n;
		}

		struct MseHist {
		public:
			int epoch;
			double mse;

			MseHist(int ep, double err) : epoch(ep), mse(err) {}
		};

	protected:

		virtual void trainOnSet(NeuralNetwork& network,
			double* inputs, double* expOutputs,
			double* buffer, double* outPtr) {}

		virtual void trainOnEpoch(NeuralNetwork& network,
			int trainingSets, double* buffer,
			double** inputSet, size_t inLength,
			double** expOutputSet, size_t outLength) {}


		virtual void initTraining(NeuralNetwork& network,
			int trainingSets,
			double** inputSet, size_t inLength,
			double** expOutputSet, size_t outLength) {}

		virtual void initTrainingEpoch(NeuralNetwork& network,
			int trainingSets,
			double** inputSet, size_t inLength,
			double** expOutputSet, size_t outLength) {}

		virtual void initTrainingSet(NeuralNetwork& network,
			double* inputs, size_t inLength,
			double* expOutputs, size_t outLength) {}

		virtual void cleanUp() {}

	public:
		SupervisedTrainer(double learnRate = 0.1, double error = 0.002, int epochs = 1000) {
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

	private:
		// used in logging mse history
		const int MSE_MAXC = 15;
		const int MSE_TRAILC = 5;

	public:
		void train(NeuralNetwork& network, int trainingSets,
			double** inputSet,	 size_t inLength,
			double** expOutputSet, size_t outLength) {
			if (network.expectedInputs() != inLength)
				throw invalid_argument("Input of network and size of input buffer don't match.");
			if (network.expectedOutputs() != outLength)
				throw invalid_argument("Output of network and size of output buffer don't match.");

			int bufferSize = network.expectedBufferSize();

			unique_ptr<double[]> bufferPtr(new double[bufferSize]);
			double* buffer = bufferPtr.get();

#ifndef FAST_MODE
			vector<MseHist> mseHistory;
			mseHistory.reserve(MSE_TRAILC * 2 + MSE_MAXC + 1);
			int mseRecordMod = (epochTarget - MSE_TRAILC) / MSE_MAXC;
			MseHist maxMse = MseHist(0, 0);
			MseHist minMse = MseHist(0, 0);
#endif

			double mse = 0;
			int e = 0;
			try {
				setError = Eigen::VectorXd(trainingSets);
				// init setError before training
				for (int i = 0; i < trainingSets; i++) {
					double* inputs = inputSet[i];
					double* expOutputs = expOutputSet[i];
					double* outPtr = executeOnSet(network, buffer,
						inputs, inLength, expOutputs, outLength);

					double setMse = cost(outLength, outPtr, expOutputSet[i]);
					setError(i) = setMse;
				}

#ifndef FAST_MODE
				mse = setError.sum() / trainingSets;
				maxMse = MseHist(-1, mse);
				minMse = MseHist(-1, mse);
#endif

				initTraining(network, trainingSets,
					inputSet, inLength, expOutputSet, outLength);

				while(e < epochTarget) {
					initTrainingEpoch(network, trainingSets,
						inputSet, inLength, expOutputSet, outLength);
					currSet = 0;

					for (int i = 0; i < trainingSets; i++) {
						double* inputs = inputSet[i];
						double* expOutputs = expOutputSet[i];
						double* outPtr = executeOnSet(network, buffer,
							inputs, inLength, expOutputs, outLength);

						double setMse = cost(outLength, outPtr, expOutputSet[i]);
						setError(i) = setMse;

						trainOnSet(network, inputs, expOutputs, buffer, outPtr);
						currSet++;
					}

					trainOnEpoch(network, trainingSets, buffer,
						inputSet, inLength, expOutputSet, outLength);

					mse = setError.sum() / trainingSets;
#ifndef FAST_MODE
					MseHist mseHist = MseHist(e, mse);
					mseHistory.push_back(mseHist);

					if (mseHist.mse > maxMse.mse) maxMse = mseHist;
					if (mseHist.mse < minMse.mse) minMse = mseHist;
#endif

					if (mse <= errorTarget) break;
					else if (mse > mseMax) break;

#ifndef FAST_MODE
					if (mseHistory.size() > MSE_TRAILC * 2 + MSE_MAXC) {
						mseHistory.erase(mseHistory.begin() + MSE_TRAILC + e / mseRecordMod);
					}
#endif

					e++;
				}
			}
			catch (exception ex) {
				printf("\n\n!!! ERROR: Threw exception while training: %s", ex.what());
				printf("\nFailed on epoch %d with MSE of %.6e", e, mse);
			}

			cleanUp();

#ifndef FAST_MODE
			displayResults(network, buffer, trainingSets,
				inputSet, inLength, expOutputSet, outLength, mse, e);

			printf("%-10s | -", "MSE Trend");
			if (!mseHistory.empty()) {
				double mseRange = maxMse.mse - minMse.mse;

				printf("\nMin [ %d:%.6e ] -> Max [ %d:%.6e ]", minMse.epoch + 1, minMse.mse, maxMse.epoch + 1, maxMse.mse);

				double prevMse = 0;
				for (int i = 0; i < mseHistory.size(); i++) {
					MseHist entry = mseHistory[i];

					const char* compare;
					if (entry.mse > prevMse)
						compare = "\x1B[31m>\033[0m";
					else if (entry.mse < prevMse)
						compare = "\x1B[32m<\033[0m";
					else
						compare = "=";

					printf("\n%-10d | [ %.6e %s ] ", entry.epoch + 1, entry.mse, compare);

					int starCount = (int)(24 * (entry.mse - minMse.mse) / mseRange);
					if ((starCount == 0 || starCount > 24) && entry.mse != minMse.mse) {
						mseRange = entry.mse - minMse.mse;
						starCount = (int)(24 * (entry.mse - minMse.mse) / mseRange);
					}
					for (int n = 0; n < starCount; n++) {
						printf("*");
					}

					prevMse = entry.mse;
				}
				printf("\n");
			}
			else {
				printf("None");
			}
#endif
		}

		void train(NeuralNetwork& network,
			double* inputs, size_t inLength, double* expOutputs, size_t outLength) {
			unique_ptr<double* []> inputSet(new double* [1] {inputs});
			unique_ptr<double* []> expOutputSet(new double* [1] { expOutputs });

			train(network,
				1,
				inputSet.get(), inLength,
				expOutputSet.get(), outLength);
		}

	protected:
		double* executeOnSet(NeuralNetwork& network,
			double* buffer,
			double* inputs,	  size_t inLength,
			double* expOutputs, size_t outLength) {

			initTrainingSet(network, inputs, inLength, expOutputs, outLength);
			memcpy(buffer, inputs, inLength * sizeof(double));
			return network.executeToIOArray(buffer, inLength, network.expectedBufferSize());
		}
	private:
		void displayResults(NeuralNetwork& network, double* buffer,
			int trainingSets,
			double** inputSet, int inLength,
			double** expOutputSet, int outLength,
			double mse, int e) {
			bool failed = false;

			for (int i = 0; i < trainingSets; i++) {
				double* inputs = inputSet[i];
				double* expOutputs = expOutputSet[i];

				printf("\n\n### Training set #%d\n", i);
				printf("\n%-10s | [ ", "Inputs");
				for (int i = 0;;) {
					printf("%6.3f", inputs[i]);

					if (++i < inLength) {
						printf(", ");
					}
					else break;
				}
				printf(" ]");

				printf("\n%-10s | [ ", "ExpOutputs");
				for (int i = 0;;) {
					printf("%9.6f", expOutputs[i]);

					if (++i < outLength) {
						printf(", ");
					}
					else break;
				}
				printf(" ]");

				try {
					double* outPtr = executeOnSet(network, buffer,
						inputs, inLength, expOutputs, outLength);

					printf("\n%-10s | [ ", "NNOutputs");
					for (int i = 0;;) {
						printf("%9.6f", outPtr[i]);

						if (++i < outLength) {
							printf(", ");
						}
						else break;
					}
					printf(" ]");
					printf("\n%-10s | [ %.6e ]\n", "MSError", cost(outLength, outPtr, expOutputs));
				}
				catch (exception ex) {
					printf("\n%-10s | [ FAILED ]", "NNOutputs");
					printf("\n%-10s | [ FAILED ]\n", "MSError");
					failed = true;
				}
			}
			printf("\n\n### Final Results");
			if (failed) {
				printf("\n%-10s | %-30s | Epoch %-3d", "Result", "[ FAILED ]", e);
			}
			else if (e == epochTarget) {
				printf("\n%-10s | %-30s | Epoch %-3d", "Result", "Failed - Reached epoch limit", e);
			}
			else if(mse > mseMax) {
				printf("\n%-10s | %-30s | Epoch %-3d", "Result", "Failed - Reached max MSE limit", e);
			} else {
				printf("\n%-10s | %-30s | Epoch %-3d", "Result", "Succeeded - Reached minimum MSE target", e);
			}
			printf("\n%-10s | [ %.6e ]\n", "MMSError", mse);
		}
	};
};