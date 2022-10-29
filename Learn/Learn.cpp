/*
* This project is based off of Chapter 7: "Machine Learning" of
* Artificial Intelligence Using C by Herbert Schildt.
*/
#include <iostream>
#include "DescriptionLearner.h"
#include "DLearnerListData.h"

#include "NeuralNetwork.h"
#include "nn/SupervisedNetworkTrainer.h"
#include "nn/PerceptronTrainer.h"

int main()
{

	string val;
	for (;;) {
		printf("Choose a program:\n");
		printf("  1: DescriptionLearner\n");
		printf("  2: NeuralNetwork\n");
		printf("  q: quit\n");
		printf("? ");

		getline(cin, val);

		if (val.size() > 0) {
			char ch = val.c_str()[0];
			switch (ch) {
			case 'q': return 0;

			case '1':
				ml::DescriptionLearner desc;
				desc.initialize(new ml::DLearnerListData());
				desc.execute();
				break;

			case '2':
				nn::NeuralNetwork net({
					nn::NeuronLayer(3, nn::Step, "in"),
					nn::NeuronLayer(1, nn::Step, "out")
				});

				double** ptronTrainingIn = new double* [4] {
					new double[3] { 1.0, 0.0, 0.0 },
					new double[3] { 1.0, 0.0, 1.0 },
					new double[3] { 1.0, 1.0, 0.0 },
					new double[3] { 1.0, 1.0, 1.0 }
				};
				double** ptronTrainingOut = new double* [4] {
					new double[1] { 0.0 },
					new double[1] { 0.0 },
					new double[1] { 0.0 },
					new double[1] { 1.0 },
				};

				printf("### NETWORK BEFORE TRAINING ###\n");
				net.display();

				nn::NeuralNetwork newNet = nn::NeuralNetwork(net);

				nn::PerceptronTrainer trainer = nn::PerceptronTrainer();
				trainer.setEpochTarget(10);

				for (int i = 0; i < 4; i++) {
					try {
						printf("\n### Training set #%s\n", to_string(i).c_str());
						trainer.trainInplace(newNet, ptronTrainingIn[i], 3, ptronTrainingOut[i], 1);
					}
					catch (exception e) {
						printf("\n\n!!! ERROR !!! Threw exception while training: %s", e.what());
						return -1;
					}
				}

				printf("### NETWORK AFTER TRAINING ###\n");
				net.displayChange(newNet);

				break;
			}
		} // if 
		printf("\n\n");
	} // for

}