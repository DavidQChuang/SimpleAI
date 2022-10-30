/*
* This project is based off of Chapter 7: "Machine Learning" of
* Artificial Intelligence Using C by Herbert Schildt.
*/
#include <iostream>
#include <chrono>
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
		printf("  1: Description Learner\n");
		printf("  2: Neural Network - Perceptron\n");
		printf("  3: Neural Network - Adaline\n");
		printf("  q: quit\n");
		printf("? ");

		getline(cin, val);

		if (val.size() > 0) {
			char ch = val.c_str()[0];
			try {
				execute(ch);
			}
			catch (exception e) {
				printf("Failed: %s", e.what());
			}
		} // if 
		printf("\n\n");
	} // for
}
void execute(char ch) {
	switch (ch) {
	case 'q':
		return;

	case '1': descriptionLearner();
		break;
	case '2': nnPerceptron();
		break;
	case '3': nnAdaline();
		break;

	} // switch (ch)
}

void descriptionLearner() {
	ml::DescriptionLearner desc;
	desc.initialize(new ml::DLearnerListData());
	desc.execute();
}

void nnPerceptron() {
	nn::NeuralNetwork net({
		nn::NeuronLayer(3, nn::ActFunc::Linear, "in"),
		nn::NeuronLayer(1, nn::ActFunc::Step, "out")
		});

	double** ptronTrainingIn = new double* [4]{
		new double[3] { 1.0, 0.0, 0.0 },
		new double[3] { 1.0, 0.0, 1.0 },
		new double[3] { 1.0, 1.0, 0.0 },
		new double[3] { 1.0, 1.0, 1.0 }
	};
	double** ptronTrainingOut = new double* [4]{
		new double[1] { 0.0 },
		new double[1] { 0.0 },
		new double[1] { 0.0 },
		new double[1] { 1.0 },
	};

	printf("### TRAINING NETWORK ###\n---------------------------\n");
	net.getLayers()[1].weightsOut()[0] = 1;

	nn::NeuralNetwork newNet = nn::NeuralNetwork(net);
	nn::PerceptronTrainer trainer = nn::PerceptronTrainer(0.01, 0.002, 100);

	auto start = chrono::high_resolution_clock::now();
	for (int i = 0; i < 4; i++) {
		try {
			printf("\n\n### Training set #%s\n", to_string(i).c_str());
			trainer.trainInplace(newNet, ptronTrainingIn[i], 3, ptronTrainingOut[i], 1);
		}
		catch (exception e) {
			printf("\n\n!!! ERROR !!! Threw exception while training: %s", e.what());
			return;
		}
	}
	auto stop = chrono::high_resolution_clock::now();

	printf("### NETWORK AFTER TRAINING ###\n---------------------------\nExec time: %fms\n",
		chrono::duration_cast<chrono::milliseconds>(stop - start).count());
	net.displayChange(newNet);
}

void nnAdaline() {
	nn::NeuralNetwork net({
		nn::NeuronLayer(3, nn::ActFunc::Linear, "in"),
		nn::NeuronLayer(1, nn::ActFunc::Step, "out")
		});

	double** ptronTrainingIn = new double* [4]{
		new double[3] { 1.0, 0.0, 0.0 },
		new double[3] { 1.0, 0.0, 1.0 },
		new double[3] { 1.0, 1.0, 0.0 },
		new double[3] { 1.0, 1.0, 1.0 }
	};
	double** ptronTrainingOut = new double* [4]{
		new double[1] { 0.0 },
		new double[1] { 0.0 },
		new double[1] { 0.0 },
		new double[1] { 1.0 },
	};

	printf("### TRAINING NETWORK ###\n---------------------------\n");
	net.getLayers()[1].weightsOut()[0] = 1;

	nn::NeuralNetwork newNet = nn::NeuralNetwork(net);
	nn::PerceptronTrainer trainer = nn::PerceptronTrainer(0.01, 0.002, 100);

	auto start = chrono::high_resolution_clock::now();
	for (int i = 0; i < 4; i++) {
		try {
			printf("\n\n### Training set #%s\n", to_string(i).c_str());
			trainer.trainInplace(newNet, ptronTrainingIn[i], 3, ptronTrainingOut[i], 1);
		}
		catch (exception e) {
			printf("\n\n!!! ERROR !!! Threw exception while training: %s", e.what());
			return;
		}
	}
	auto stop = chrono::high_resolution_clock::now();

	printf("### NETWORK AFTER TRAINING ###\n---------------------------\nExec time: %fms\n",
		chrono::duration_cast<chrono::milliseconds>(stop - start).count());
	net.displayChange(newNet);
}
