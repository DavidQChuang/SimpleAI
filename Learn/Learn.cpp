/*
* This project is based off of Chapter 7: "Machine Learning" of
* Artificial Intelligence Using C by Herbert Schildt.
*/
#include <iostream>
#include "DescriptionLearner.h"
#include "DLearnerListData.h"

#include "NeuralNetwork.h"
#include "nn/NeuralNetworkTrainer.h"

int main()
{
	printf("Choose a program:\n");
	printf("  1: DescriptionLearner\n");
	printf("  2: NeuralNetwork\n");
	printf("? ");

	string val;
	for (;;) {
		getline(cin, val);

		if (val.size() > 0) {
			char ch = val.c_str()[0];
			switch (ch) {
			case '1':
				ml::DescriptionLearner desc;
				desc.initialize(new ml::DLearnerListData());
				desc.execute();
				break;
			case '2':
				nn::NeuronLayer input(2, "in");
				nn::NeuronLayer layer1(3, "layer1");
				nn::NeuronLayer layer2(3, "layer2");
				nn::NeuronLayer output(1, "out");

				vector<nn::NeuronLayer> layers = { input, layer1, layer2, output };
				nn::NeuralNetwork net(layers);

				nn::SupervisedNetworkTrainer trainer;
				//trainer.epochs

				break;
			}
		}
	}

}