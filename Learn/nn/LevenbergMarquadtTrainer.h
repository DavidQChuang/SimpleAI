#pragma once

#include "SupervisedTrainer.h"
#include <Eigen/Dense>

namespace nn {
	// Useful: https://www.codeproject.com/articles/55691/neural-network-learning-by-the-levenberg-marquardt
	// https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm#Choice_of_damping_parameter
	class LevenbergMarquadtTrainer : public SupervisedTrainer {
	private:
		int jacobianCols = 0;
		int jacobianRows = 0;
		Eigen::MatrixXd J;
		Eigen::VectorXd error;

		int set = 0;

		//double learningRate

		double learningRateFunc() {
			return learningRate;
		}

	protected:

		void initTrainingEpoch(NeuralNetwork& network, int trainingSets,
			double** inputSet, size_t inLength, double** expOutputSet, size_t outLength) {

			jacobianCols = 0;
			auto layers = network.getLayers();
			for (int l = 0; l < layers.size(); l++) {
				auto layer = layers[l];
				jacobianCols += layer.size() * layer.inputsPerNeuron();
			}

			jacobianRows = trainingSets;

			J = Eigen::MatrixXd(jacobianRows, jacobianCols);
			error = Eigen::VectorXd(jacobianRows);

			set = 0;
		}

		void initTrainingSet(NeuralNetwork& network,
			double* inputs, size_t inLength, double* expOutputs, size_t outLength) override {
			SupervisedTrainer::initTrainingSet(network, inputs, inLength, expOutputs, outLength);
		}

		void cleanUp() override {
		}

		void trainOnSet(NeuralNetwork& network, double* inputs, double* expOutputs, double* buffer, double* outPtr) {
			vector<double> layerDelta;
			vector<double> oldLayerDelta;
			vector<double> errors;
			vector<NeuronLayer>& layers = network.getLayers();

			double errorMean = 0;

			// Calculate target vs. nn output errors and store them in the errors buffer.
			// The error cancels out for the output layer in the Levenberg-Marquadt equation,
			// so the the delta of the output layer will just be 1 * f'(hi) instead of e * f'(hi).
			int out = 0;
			NeuronLayer& outputLayer = layers[layers.size() - 1];
			for (int n = 0; n < outputLayer.size(); n++) {
				double sumNN = 0;
				double sumExp = 0;

				for (int o = 0; o < outputLayer.outputsPerNeuron(); o++) {
					sumNN += outPtr[out];
					sumExp += expOutputs[out];
					out++;
				}

				double error = sumExp - sumNN;

				errorMean += pow(error, 2);
				errors.push_back(sumExp - sumNN);
				layerDelta.push_back(1);
			}
			errorMean /= outputLayer.size();
			error(set) = errorMean;

			double* inPtr = outPtr;

			int layerWeightIndex = jacobianCols;
			// Update the jacobian matrix using the same deltas from normal backpropagation.
			for (int l = layers.size() - 1; l >= 0; l--) {
				NeuronLayer& layer = layers[l];
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
						J(set, layerWeightIndex + w) = delta * inPtr[in] / errorMean;
						in++;
					}

					// If the inputs for this layer's neurons are independent,
					// the inputs overlap instead of being stored sequentially.
					if (!layer.independentInputs()) {
						in = 0;
					}
				}
			} // for

			set++;
		}
		
		void trainOnEpoch(NeuralNetwork& network, int trainingSets,
			double** inputSet, size_t inLength, double** expOutputSet, size_t outLength) {
			set = 0;
			// delta W = (JTJ + LI)^-1JT (Y - f(X, W))

			//std::cout << "TEST J: " << J << std::endl;
			//std::cout << "TEST ERROR: " << error << std::endl;

			auto F1A = J.transpose() * J; // square matrix, size nxn
			auto F1 = F1A + Eigen::MatrixXd::Identity(jacobianCols, jacobianCols) * learningRate; // square matrix, size nxn
			auto F2 = J.transpose() * error; // nxm * mx1 = nx1: # of weights x
			auto F = F1.inverse() * F2; // nxn * nx1: nx1

			//std::cout << "TEST F: " << F << std::endl;
			
			vector<NeuronLayer>& layers = network.getLayers();
			int n = 0;
			int nMax = layers[0].size() - 1;
			int l = 0;
			vector<double>& weightsIn = layers[0].weightsIn();
			for (int i = 0; i < F.cols(); i++) {
				if (n > nMax) {
					n = 0;
					weightsIn = layers[++l].weightsIn();
				}

				weightsIn[n++] += F[i];
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