#pragma once

#include <vector>
#include <string>

#include "nn/NeuronLayer.h"

namespace nn {
	/// <summary>
	/// Fully connected feedforward network, also known as 'multilayer perceptrons'.
	/// </summary>
	/// <typeparam name="...LayerArgs">The concrete types of the layers of this network.</typeparam>
	template<typename... LayerArgs>
	class FFNeuralNetwork {
	public:
		typedef INeuronLayer Layer;

	private:
		static_assert((std::is_base_of<INeuronLayer, LayerArgs>::value && ...),
			"Arguments must be derived from INeuronLayer.");

		typedef std::vector<INeuronLayer*> NNLayers;
		typedef std::tuple<LayerArgs...> NNLayerTuple;

		// fields
		std::vector<INeuronLayer*> nnLayers;
		std::tuple<LayerArgs...> nnLayerTuple;

		int ioBufferSize;

		int inputs = 0;
		int outputs = 0;

		template<std::size_t... Is>
		void initLayerVector(std::index_sequence<Is...>) {
			auto add = [&](auto& layer) {
				nnLayers.push_back(&layer);
			};

			(add(std::get<Is>(nnLayerTuple)), ...);
		}

	public:
		FFNeuralNetwork(std::tuple<LayerArgs...> layerArgs)
			: nnLayerTuple(layerArgs) {
			constexpr size_t layerCount = std::tuple_size_v<NNLayerTuple>;
			initLayerVector(std::make_index_sequence<layerCount>{});

			if (layerCount == 0) {
				throw invalid_argument("The neural network cannot have zero layers.");
			}

			int prevOutputs = 1;
			for (int i = 0; i < layerCount; i++) {
				Layer& layer = *nnLayers[i];

				// fully connected
				// inputs per neuron = output count of prev layer,
				// outputs per neuron = 1, input layer is 1 to 1 (independent),
				// the hidden & output layers are [prev outputs] to 1 (shared).
				bool indepInputs = (i == 0);
				layer.init(prevOutputs, 1, indepInputs, true);

				// check that this layer has the properties above
				if (Layer::totalInputs(layer.size(), prevOutputs, indepInputs) != layer.totalInputs())
					throw out_of_range("Layers have incompatible in/out sizes.");

				prevOutputs = layer.totalOutputs();
				ioBufferSize += layer.totalInputs();
			}

			inputs = (*nnLayers[0]).totalInputs();
			outputs = (*nnLayers[layerCount - 1]).totalOutputs();

			ioBufferSize += outputs;
		}

	public:
		inline int expectedInputs() const { return inputs; }
		inline int expectedOutputs() const { return outputs; }

		inline int expectedBufferSize() const { return ioBufferSize; }

		inline Layer& getLayer(int i) {
			return *nnLayers[i];
		}

		inline int depth() const {
			return nnLayers.size();
		}

		double* execute(double* inputs, size_t inLength) {
			double* buffer = new double[ioBufferSize];
			memcpy(buffer, inputs, inLength * sizeof(double));

			return executeToIOArray(buffer, inLength, ioBufferSize);
		}

		double* executeToIOArray(double* buffer, size_t inLength, size_t bufferSize) {
			if (inLength != (*nnLayers[0]).totalInputs())
				throw invalid_argument("Expected input size did not match given input size.");

			if (ioBufferSize != bufferSize)
				throw invalid_argument("Expected buffer size did not match given buffer size.");

			constexpr size_t size = std::tuple_size_v<NNLayerTuple>;
			executeLayers(buffer, std::make_index_sequence<size>{});

			return buffer + (ioBufferSize - (*nnLayers.back()).totalOutputs());
		}

	private:
		template<std::size_t... Is>
		void executeLayers(double* buffer, std::index_sequence<Is...>) {
			double* inPtr = buffer;
			auto exec = [&inPtr, &buffer](auto& layer) {
				int inLen = layer.totalInputs();
				int outLen = layer.totalOutputs();

				double* outPtr = inPtr + inLen;

				layer.execute(inPtr, inLen, outPtr, outLen);
				inPtr = outPtr;
			};

			(exec(std::get<Is>(nnLayerTuple)), ...);
		}

	public:
		void display() {
			printf("\nExpected in/out: %s/%s\n", to_string(inputs).c_str(), to_string(outputs).c_str());

			if (nnLayers.size() == 1) {
				printf("### Input Layer");
				(*nnLayers[0]).display();
			}
			else if (nnLayers.size() >= 2) {
				NNLayers::iterator it;

				printf("### Input Layer");
				(*nnLayers.front()).display();
				printf("\n");

				for (it = nnLayers.begin() + 1; it < nnLayers.end() - 1; it++) {
					Layer& layer = **it;

					printf("\n### Hidden Layer - %s", layer.name().c_str());
					layer.display();
					printf("\n");
				}

				printf("\n### Output Layer");
				(*nnLayers.back()).display();
			}
		}

		void displayChange(FFNeuralNetwork<LayerArgs...> other) {
			printf("Expected in/out: %s/%s\n", to_string(inputs).c_str(), to_string(outputs).c_str());

			if (nnLayers.size() == 1) {
				printf("### Input Layer");
				(*nnLayers[0]).display();
				(*other.nnLayers[0]).display();
			}
			else if (nnLayers.size() >= 2) {
				NNLayers::iterator it, it2;

				printf("### Input Layer");
				(*nnLayers.front()).display();
				(*other.nnLayers.front()).display();
				printf("\n");

				for (it = nnLayers.begin() + 1, it2 = other.nnLayers.begin() + 1; it < nnLayers.end() - 1; it++, it2++) {
					Layer& layer1 = **it;
					Layer& layer2 = **it2;

					printf("\n### Hidden Layer - %s", layer1.name().c_str());
					layer1.display();
					printf("\n### Hidden Layer - %s", layer2.name().c_str());
					layer2.display();
					printf("\n");
				}

				printf("\n### Output Layer");
				(*nnLayers.back()).display();
				(*other.nnLayers.back()).display();
			}
		}
	};

	struct NeuralNetwork {
	public:
		using Layer = INeuronLayer;

		template<template<class...> class T, class U>
		struct is_template_of
		{
			template<class... TT>
			static std::true_type test(T<TT...>*);

			static std::false_type test(...);

			constexpr static bool value = decltype(test((U*)nullptr)){};
		};

		template<template<class...> class T, class...>
		struct is_template_of_N : std::true_type
		{};

		template<template<class...> class T, class U, class... TT>
		struct is_template_of_N<T, U, TT...>
			: std::integral_constant<bool, is_template_of<T, U>::value
			&& is_template_of_N<T, TT...>{} >
		{};

		template<typename... LayerArgs>
		static typename std::enable_if<
			!is_template_of_N<std::tuple, LayerArgs...>::value &&
			(std::is_base_of<INeuronLayer, LayerArgs>::value && ...),
			nn::template FFNeuralNetwork<LayerArgs...>
		>::type	MakeNetwork(LayerArgs... layerArgs)
		{
			return FFNeuralNetwork<LayerArgs...>(std::tuple(layerArgs));
		}

		template<typename... LayerArgs>
		static typename std::enable_if<
			(std::is_base_of<INeuronLayer, LayerArgs>::value && ...),
			nn::template FFNeuralNetwork<LayerArgs...>
		>::type MakeNetwork(std::tuple<LayerArgs...> layerArgs)
		{
			return FFNeuralNetwork<LayerArgs...>(layerArgs);
		}

		template<template<class...> class Trainer, typename... LayerArgs, typename... TrainerArgs>
		static typename std::enable_if<
			(std::is_base_of<INeuronLayer, LayerArgs>::value && ...),
			Trainer<LayerArgs...>
		>::type MakeTrainer(std::tuple<LayerArgs...>, TrainerArgs... tArgs) {
			return Trainer<LayerArgs...>(tArgs...);
		}

	private:

	};
}