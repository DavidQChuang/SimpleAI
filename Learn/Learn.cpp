/*
* This project is based off of Chapter 7: "Machine Learning" of
* Artificial Intelligence Using C by Herbert Schildt.
*/
#include <iostream>
#include "DescriptionLearner.h"
#include "DLearnerListData.h"
#include "NeuralNetwork.h"

int main()
{
	printf("Choose a program:\n");
	printf("\t1: DescriptionLearner");
	printf("\t2: NeuralNetwork");
	printf("\n");

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
				vector<nn::NeuronLayer> hiddenLayers{
					nn::NeuronLayer(3, "layer 1"),
					nn::NeuronLayer(3, "layer 2"),
				};
				nn::NeuronLayer output(1, "out");

				nn::NeuralNetwork net(&input, &output, hiddenLayers);

				break;
			}
		}
	}

}