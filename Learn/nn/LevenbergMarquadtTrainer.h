#pragma once

#include "SupervisedTrainer.h"
#include <Eigen/Dense>

namespace nn {
	// Useful: https://www.codeproject.com/articles/55691/neural-network-learning-by-the-levenberg-marquardt
	// https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm#Choice_of_damping_parameter
	class LevenbergMarquadtTrainer : public SupervisedTrainer {
	private:
		// fields set & used during training
		// persistent for all epochs
		int jacobianCols = 0;
		int jacobianRows = 0;
		Eigen::MatrixXd J;
		Eigen::VectorXd Wd;

		// variable per epoch
		double dampingFactor = 0.1;

		// constant for all epochs
		double adjustmentFactor = 10;

		// 
		double prevMse = 0;

		//double learningRate

		double learningRateFunc() {
			return learningRate;
		}

	protected:
		void initTraining(NeuralNetwork& network, int trainingSets,
			const double** inputSet, size_t inLength,
			const double** expOutputSet, size_t outLength)
		override {
			SupervisedTrainer::initTraining(network, trainingSets,
				inputSet, inLength, expOutputSet, outLength);

			jacobianCols = 0;
			for (int l = 0; l < network.depth(); l++) {
				NeuralNetwork::Layer& layer = network.getLayer(l);
				jacobianCols += layer.size() * layer.inputsPerNeuron();
			}

			jacobianRows = trainingSets;

			J = Eigen::MatrixXd(jacobianRows, jacobianCols);

			dampingFactor = learningRate;
			
			prevMse = setError.sum() / trainingSets;
		}

		/*void initTrainingEpoch(NeuralNetwork& network, int trainingSets,
			double** inputSet, size_t inLength, double** expOutputSet, size_t outLength) 
		override {
			SupervisedTrainer::initTrainingEpoch(network, trainingSets,
				inputSet, inLength, expOutputSet, outLength);
		}*/

		void cleanUp() override {}

		void trainOnSet(NeuralNetwork& network,
			const double* inputs, const double* expOutputs,
			double* buffer, double* outPtr)
		override {
			vector<double> layerDelta;
			vector<double> oldLayerDelta;

			// Calculate target vs. nn output errors and store them in the errors buffer.
			// The error cancels out for the output layer in the Levenberg-Marquadt equation,
			// so the the delta of the output layer will just be 1 * f'(hi) instead of e * f'(hi).
			int out = 0;
			NeuralNetwork::Layer& outputLayer = network.getLayer(network.depth() - 1);
			for (int n = 0; n < outputLayer.size(); n++) {
				layerDelta.push_back(1);
			}

			double* inPtr = outPtr;

			int layerWeightIndex = jacobianCols;
			// Update the jacobian matrix using the same deltas from normal backpropagation.
			for (int l = network.depth() - 1; l >= 0; l--) {
				NeuralNetwork::Layer& layer = network.getLayer(l);
				vector<double>& weightsIn = layer.weightsIn();

				inPtr -= layer.expectedInputs();

				int inputCount = layer.inputsPerNeuron();
				int in = 0;

				// Store current layer deltas and reserve and clear the next layer to 0.
				oldLayerDelta = layerDelta;
				layerDelta.reserve(inputCount);
				for (int i = 0; i < inputCount; i++) {
					if (i >= layerDelta.size()) {
						layerDelta.push_back(0);
					}
					else {
						layerDelta[i] = 0;
					}
				}

				layerWeightIndex -= inputCount * layer.size();
				for (int n = 0; n < layer.size(); n++) {
					double weightedSum = 0;

					// Sum weighted inputs of this layer - this is used later
					for (int i = 0; i < inputCount; i++) {
						int w = n * inputCount + i;

						weightedSum += inPtr[in] * weightsIn[w];
						in++;
					}
					in -= inputCount;

					// The delta for this neuron will have been calculated previously -
					// error for output layer, sum of deltas for hidden/input layers,
					// and is then multiplied by f'(h), where h is the weighted sum of inputs.
					double delta = oldLayerDelta[n] * layer.derivActivationFunc(weightedSum);

					// Adjust the weights for each neuron in sequence.
					for (int i = 0; i < inputCount; i++) {
						int w = n * inputCount + i;

						// Each input corresponds to a neuron in the preceding layer.
						// The next layer's delta for that neuron [i] is the sum of this
						// layer's neurons' deltas dj * the weight wij connecting the two
						// neurons for each neuron [j] in this layer.
						layerDelta[i] += delta * weightsIn[w];
						J(currSet, layerWeightIndex + w) = delta * inPtr[in] / setError(currSet);
						in++;
					}

					// If the inputs for this layer's neurons are independent,
					// the inputs overlap instead of being stored sequentially.
					if (!layer.independentInputs()) {
						in = 0;
					}
				}
			} // for
		}
		
		void trainOnEpoch(NeuralNetwork& network, int trainingSets, double* buffer,
			double** inputSet, size_t inLength, double** expOutputSet, size_t outLength) {
			// delta W = (JTJ + LI)^-1JT (Y - f(X, W))

			//std::cout << "TEST J: " << J << std::endl;
			//std::cout << "TEST ERROR: " << error << std::endl;

			auto F1A = J.transpose() * J; // square matrix, size nxn
			auto F1 = F1A + Eigen::MatrixXd::Identity(jacobianCols, jacobianCols) * dampingFactor; // square matrix, size nxn
			auto F2 = J.transpose() * setError; // nxm * mx1 = nx1: # of weights x
			Wd = F1.inverse() * F2; // nxn * nx1: nx1

			//std::cout << "TEST F: " << F << std::endl;
			
			updateWeights<1>(network, Wd);
			/*
			// Recalculate MSE after weight update
			for (int i = 0; i < trainingSets; i++) {
				double* inputs = inputSet[i];
				double* expOutputs = expOutputSet[i];
				double* outPtr = executeOnSet(network, buffer,
					inputs, inLength, expOutputs, outLength);

				double setMse = cost(outLength, outPtr, expOutputSet[i]);
				setError(i) = setMse;
			}

			double mse = setError.sum() / trainingSets;
			if (mse >= prevMse) {
				// Failed to reduce mse. Discard weights and increase damping factor.
				dampingFactor *= adjustmentFactor;
				updateWeights<-1>(network, Wd);
			}
			else if (mse < prevMse) {
				// Reduced mse successfully. Keep weights and reduce damping factor.
				dampingFactor /= adjustmentFactor;
			}
			prevMse = mse;*/
		}

	private:
		template<int factor>
		void updateWeights(NeuralNetwork& network, Eigen::VectorXd F) {
			int l = 0;
			vector<double>* weightsIn = &network.getLayer(0).weightsIn();
			int w = 0;
			int wMax = weightsIn->size();
			for (int i = 0; i < F.size(); i++) {
				if (w >= wMax) {
					w = 0;
					weightsIn = &network.getLayer(++l).weightsIn();
					wMax = weightsIn->size();
				}

				double dW = F[i];
				(*weightsIn)[w++] += factor * dW;
			}
		}

	public:
		//LevenbergMarquadtTrainer(double learnRate = 0.1, double error = 0.002, int epochs = 1000)
		//	: LevenbergMarquadtTrainer(learnRate, error, epochs, learnRate) {}
		//LevenbergMarquadtTrainer(double learnRate, double error, int epochs, double dampingFactor)
		//	: SupervisedTrainer(learnRate, error, epochs), dampingFactor(dampingFactor) { }


		LevenbergMarquadtTrainer(double learnRate = 0.1, double error = 0.002, int epochs = 1000)
			: SupervisedTrainer(learnRate, error, epochs) {}
	};
}