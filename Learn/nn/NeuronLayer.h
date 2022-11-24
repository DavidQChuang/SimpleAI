#pragma once

#include <vector>
#include <random>
#include <string>
#include <stdexcept>
#include <functional>
#include "../statmath.h"

namespace nn {
	enum class ScalarFunc {
		Step, Linear, Siglog, Hypertan, ReLU, LeakyReLU, GeLU,
	};

	enum class VectorFunc {
		Softmax, Argmax
	};

	enum class WeightInit {
		Constant, Normal, Uniform
	};

	class INeuronLayer {
	protected:
		INeuronLayer() {}

		int		mNeuronInputs = 0; // input per neuron
		int		mNeuronOutputs = 0; // output per neuron
		bool	mUseInputs = true;
		bool	mIndependentInputs = false;
		int		neuronCount = 0;

		std::vector<double> inputWeights;

		bool overrideUseInputs = false;
		bool overrideIndependentInputs = false;

		std::string layerName = "";

	public:
		INeuronLayer(int count, std::string name = "Layer") {
			if (count == 0) throw std::out_of_range("Invalid neuron count, cannot be zero.");

			neuronCount = count;
			layerName = name;
		}

		INeuronLayer(int count, bool independentInputs, bool useInputs, std::string name = "Layer")
			: INeuronLayer(count, name) {
			overrideUseInputs = true;
			overrideIndependentInputs = true;

			mUseInputs = useInputs;
			mIndependentInputs = independentInputs;
		}

		virtual INeuronLayer* clone() = 0;

		void init(int inputsPerNeuron, int outputsPerNeuron, bool independentInputs, bool useInputs);

		template<WeightInit initType, typename... Args>
		void initWeights(Args... args) = delete;

		template<> void initWeights<WeightInit::Constant, double>(double weight);
		template<> void initWeights<WeightInit::Normal, double, double>(double stdev, double mean);
		template<> void initWeights<WeightInit::Uniform, double, double>(double min, double max);

		virtual void execute(double* input, int inputLength, double* output, int outputLength);

		virtual void display();

		virtual double activationFunc(double v) = 0;
		virtual double derivActivationFunc(double v) = 0;
		virtual void vectorActivationFunc(double* output, int outputLength) {}

	public:
		////////////////////////
		// GETTERS
		inline int expectedInputs() { // if inputs are independent, they don't overlap
			return mNeuronInputs * (mIndependentInputs ? neuronCount : 1);
		}
		inline int expectedOutputs() { return mNeuronOutputs * neuronCount; }

		inline int inputsPerNeuron() { return mNeuronInputs; }
		inline int outputsPerNeuron() { return mNeuronOutputs; }

		inline int size() { return neuronCount; }
		inline std::string name() { return layerName; }

		inline bool independentInputs() { return mIndependentInputs; }

		inline std::vector<double>& weightsIn() { return inputWeights; }
	};

	///////////////////////////////////////////
	/// WEIGHTED FEEDFORWARD LAYERS

	template<ScalarFunc Func>
	class FFNeuronLayer : public INeuronLayer {
	public:
		double activationFunc(double v) override = 0;
		double derivActivationFunc(double v) override = 0;
	};

#define DEFINE_VLAYER(className) \
	template<>\
	class className : public INeuronLayer {\
	public:\
		className(int count, std::string name = "Layer")\
			: INeuronLayer(count, name) {}\
		className(int count, bool independentInputs, bool useInputs, std::string name = "Layer")\
			: INeuronLayer(count, independentInputs, useInputs, name) {}\
	\
		INeuronLayer* clone() override { return new className(*this); }\
	\
		double activationFunc(double v) override;\
		double derivActivationFunc(double v) override;

	DEFINE_VLAYER(FFNeuronLayer<ScalarFunc::Step>)};
	DEFINE_VLAYER(FFNeuronLayer<ScalarFunc::Linear>)};
	DEFINE_VLAYER(FFNeuronLayer<ScalarFunc::Siglog>)};
	DEFINE_VLAYER(FFNeuronLayer<ScalarFunc::Hypertan>)};
	DEFINE_VLAYER(FFNeuronLayer<ScalarFunc::ReLU>)};
	DEFINE_VLAYER(FFNeuronLayer<ScalarFunc::LeakyReLU>)};
	DEFINE_VLAYER(FFNeuronLayer<ScalarFunc::GeLU>)};

	///////////////////////////////////////////
	/// FEEDFORWARD FILTER LAYERS

	template<VectorFunc Func>
	class FFVNeuronLayer : public INeuronLayer {
	public:
		double activationFunc(double v) override { return v; }
		double derivActivationFunc(double v) override { return 1; }

		void vectorActivationFunc(double* output, int outputLength) override = 0;
	};

#define DEFINE_SLAYER(className) \
	template<>\
	class className : public INeuronLayer {\
	public:\
		className(int count, std::string name = "Layer")\
			: INeuronLayer(count, false, false, name) {}\
	\
		INeuronLayer* clone() override { return new className(*this); }\
	\
		void vectorActivationFunc(double* output, int outputLength) override;\

	DEFINE_SLAYER(FFVNeuronLayer<VectorFunc::Softmax>)
	private:
		std::vector<double> weightedSums;
		double totalSum = 0;

	public:
		double activationFunc(double v) override;
		double derivActivationFunc(double v) override;
	};

	DEFINE_SLAYER(FFVNeuronLayer<VectorFunc::Argmax>)
		double activationFunc(double v) override { return v; }
		double derivActivationFunc(double v) override { return 1; }
	};
}