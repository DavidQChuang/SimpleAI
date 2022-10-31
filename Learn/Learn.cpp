/*
* This project is based off of
* Description Learner: Herbert Schildt - Chapter 7: "Machine Learning", Artificial Intelligence Using C
* Neural Network: Alan Souza, Fábio Soares - Neural Network Programming with Java (2016)
*/

#include <iostream>
#include <chrono>
#include "DescriptionLearner.h"
#include "ml/DLearnerListData.h"

#include "NeuralNetwork.h"
#include "nn/PerceptronTrainer.h"
#include "nn/AdalineTrainer.h"

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

	constexpr int TRAINING_SETS = 4;
	constexpr int INPUTS = 3;
	constexpr int OUTPUTS = 1;

	double** trainingIn = new double* [TRAINING_SETS]{
		new double[INPUTS] { 1.0, 0.0, 0.0 },
		new double[INPUTS] { 1.0, 0.0, 1.0 },
		new double[INPUTS] { 1.0, 1.0, 0.0 },
		new double[INPUTS] { 1.0, 1.0, 1.0 }
	};
	double** trainingOut = new double* [TRAINING_SETS]{
		new double[OUTPUTS] { 0.0 },
		new double[OUTPUTS] { 0.0 },
		new double[OUTPUTS] { 0.0 },
		new double[OUTPUTS] { 1.0 },
	};

	printf("### TRAINING NETWORK ###\n---------------------------\n");
	net.getLayers()[1].weightsOut()[0] = 1;

	nn::NeuralNetwork newNet = nn::NeuralNetwork(net);
	nn::PerceptronTrainer trainer = nn::PerceptronTrainer(0.01, 0.002, 100);

	auto start = chrono::high_resolution_clock::now();
	for (int i = 0; i < TRAINING_SETS; i++) {
		try {
			printf("\n\n### Training set #%s\n", to_string(i).c_str());
			trainer.trainInplace(newNet, trainingIn[i], INPUTS, trainingOut[i], OUTPUTS);
		}
		catch (exception e) {
			printf("\n\n!!! ERROR !!! Threw exception while training: %s", e.what());
			return;
		}
	}
	auto stop = chrono::high_resolution_clock::now();

	printf("### NETWORK AFTER TRAINING ###\n---------------------------\nExec time: %lldus\n",
		chrono::duration_cast<chrono::microseconds>(stop - start).count());
	net.displayChange(newNet);
}

void nnAdaline() {
	nn::NeuralNetwork net({
		nn::NeuronLayer(4, nn::ActFunc::Linear, "in"),
		nn::NeuronLayer(1, nn::ActFunc::Linear, "out")
		});

	constexpr int TRAINING_SETS = 7;
	constexpr int INPUTS = 4;
	constexpr int OUTPUTS = 1;

	double** trainingIn = new double*[TRAINING_SETS] {
		new double[INPUTS] { 1.0, 0.98, 0.94, 0.95 },
		new double[INPUTS] { 1.0, 0.60, 0.60, 0.85 },
		new double[INPUTS] { 1.0, 0.35, 0.15, 0.15 },
		new double[INPUTS] { 1.0, 0.25, 0.30, 0.98 },
		new double[INPUTS] { 1.0, 0.75, 0.85, 0.91 },
		new double[INPUTS] { 1.0, 0.43, 0.57, 0.87 },
		new double[INPUTS] { 1.0, 0.05, 0.06, 0.01 }
	};
	double** trainingOut = new double*[TRAINING_SETS] {
		new double[1] { 0.80 },
		new double[1] { 0.59 },
		new double[1] { 0.24 },
		new double[1] { 0.45 },
		new double[1] { 0.74 },
		new double[1] { 0.63 },
		new double[1] { 0.10 },
	};

	printf("### TRAINING NETWORK ###\n---------------------------\n");
	net.getLayers()[1].weightsOut()[0] = 1;

	nn::NeuralNetwork newNet = nn::NeuralNetwork(net);
	nn::AdalineTrainer trainer = nn::AdalineTrainer(0.3, 0.000001, 100);

	auto start = chrono::high_resolution_clock::now();
	for (int i = 0; i < TRAINING_SETS; i++) {
		try {
			printf("\n\n### Training set #%s\n", to_string(i).c_str());
			trainer.trainInplace(newNet, trainingIn[i], INPUTS, trainingOut[i], OUTPUTS);
		}
		catch (exception e) {
			printf("\n\n!!! ERROR !!! Threw exception while training: %s", e.what());
			return;
		}
	}
	auto stop = chrono::high_resolution_clock::now();

	printf("### NETWORK AFTER TRAINING ###\n---------------------------\nExec time: %lldus\n",
		chrono::duration_cast<chrono::microseconds>(stop - start).count());
	net.displayChange(newNet);
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