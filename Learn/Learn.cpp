/*
* This project is based off of
* Description Learner: Herbert Schildt - Chapter 7: "Machine Learning", Artificial Intelligence Using C
* Neural Network: Alan Souza, Fábio Soares - Neural Network Programming with Java (2016)
*/

#include <iostream>
#include <chrono>
#include "DescriptionLearner.h"
#include "ml/DLearnerListData.h"

#include "nn/BackpropagationTrainer.h"
#include "NeuralNetwork.h"
#include "nn/PerceptronTrainer.h"
#include "nn/AdalineTrainer.h"

void descriptionLearner() {
	ml::DescriptionLearner desc;
	desc.initialize(new ml::DLearnerListData());
	desc.execute();
}

#define TRAINING_IN(name) double** name = new double* [TRAINING_SETS]

#define INPUT new double[INPUTS]
#define OUTPUT new double[OUTPUTS]

#define DELETE_TRAINING_DATA(n) for(int _i = 0; _i < TRAINING_SETS; _i++) \
{ delete[] n[_i]; } delete[] n; n = nullptr;

template<class T>
inline void trainNN(nn::NeuralNetwork& net, T trainer, 
	int TRAINING_SETS, double** trainingIn, int INPUTS, double** trainingOut,
	int OUTPUTS) {

	printf("### TRAINING NETWORK ###\n---------------------------\n");

	nn::NeuralNetwork newNet = nn::NeuralNetwork(net);

	auto start = chrono::high_resolution_clock::now();
	try {
		trainer.train(newNet, TRAINING_SETS, trainingIn, INPUTS, trainingOut, OUTPUTS);
	}
	catch (exception e) {
		printf("\n\n!!! ERROR !!! Threw exception while training: %s", e.what());
		return;
	}
	auto stop = chrono::high_resolution_clock::now();

	printf("### NETWORK AFTER TRAINING ###\n---------------------------\nExec time: %lldus\n",
		chrono::duration_cast<chrono::microseconds>(stop - start).count());
	net.displayChange(newNet);
}

// Training a single-layer perceptron to emulate an AND operation.
// The training algorithm used simply adjusts weights in the direction of error.
void nnPerceptron() {
	nn::NeuralNetwork net({
		nn::NeuronLayer(3, nn::ActFunc::Linear, "in"),
		nn::NeuronLayer(1, nn::ActFunc::Step, "out")
		});

	constexpr int TRAINING_SETS = 4;
	constexpr int INPUTS = 3;
	constexpr int OUTPUTS = 1;

	double** trainingIn = new double* [TRAINING_SETS]{
		INPUT { 1.0, 0.0, 0.0 },
		INPUT { 1.0, 0.0, 1.0 },
		INPUT { 1.0, 1.0, 0.0 },
		INPUT { 1.0, 1.0, 1.0 }
	};
	double** trainingOut = new double* [TRAINING_SETS]{
		OUTPUT { 0.0 },
		OUTPUT { 0.0 },
		OUTPUT { 0.0 },
		OUTPUT { 1.0 },
	};

	net.getLayers()[1].weightsIn()[0] = 1;

	nn::PerceptronTrainer trainer = nn::PerceptronTrainer(0.01, 0.002, 100);

	trainNN(net, trainer, TRAINING_SETS, trainingIn, INPUTS, trainingOut, OUTPUTS);

	DELETE_TRAINING_DATA(trainingIn);
	DELETE_TRAINING_DATA(trainingOut);
}

// Training a single-layer perceptron to solve a regression problem.
// The inputs are the amount of traffic on side roads, and the neural network
// is used to determine the resulting amount of traffic on the main avenue.
// The training algorithm used adjusts weights in the direction of error times 
// the derivative of the activation function.
void nnAdaline() {
	nn::NeuralNetwork net({
		nn::NeuronLayer(4, nn::ActFunc::Linear, "in"),
		nn::NeuronLayer(1, nn::ActFunc::Linear, "out")
		});

	constexpr int TRAINING_SETS = 7;
	constexpr int INPUTS = 4;
	constexpr int OUTPUTS = 1;

	double** trainingIn = new double*[TRAINING_SETS] {
		INPUT { 1.0, 0.98, 0.94, 0.95 },
		INPUT { 1.0, 0.60, 0.60, 0.85 },
		INPUT { 1.0, 0.35, 0.15, 0.15 },
		INPUT { 1.0, 0.25, 0.30, 0.98 },
		INPUT { 1.0, 0.75, 0.85, 0.91 },
		INPUT { 1.0, 0.43, 0.57, 0.87 },
		INPUT { 1.0, 0.05, 0.06, 0.01 }
	};
	double** trainingOut = new double*[TRAINING_SETS] {
		OUTPUT { 0.80 },
		OUTPUT { 0.59 },
		OUTPUT { 0.24 },
		OUTPUT { 0.45 },
		OUTPUT { 0.74 },
		OUTPUT { 0.63 },
		OUTPUT { 0.10 },
	};

	net.getLayers()[1].weightsIn()[0] = 1;

	nn::BackpropagationTrainer trainer = nn::BackpropagationTrainer(0.3, 1e-12, 100);

	trainNN(net, trainer, TRAINING_SETS, trainingIn, INPUTS, trainingOut, OUTPUTS);

	DELETE_TRAINING_DATA(trainingIn);
	DELETE_TRAINING_DATA(trainingOut);
}

// Training a multi-layer perceptron to solve a regression problem.
// The inputs are ...
// The training algorithm used adjusts weights ...
void nnBackpropagation() {
	nn::NeuralNetwork net({
		nn::NeuronLayer(3, nn::ActFunc::Siglog, "in"),
		nn::NeuronLayer(3, nn::ActFunc::Siglog, "hidden"),
		nn::NeuronLayer(2, nn::ActFunc::Linear, "out")
		});

	constexpr int TRAINING_SETS = 10;
	constexpr int INPUTS = 3;
	constexpr int OUTPUTS = 2;

	double** trainingIn = new double* [TRAINING_SETS] {
		INPUT{ 1.0, 1.0, 0.73 },
		INPUT{ 1.0, 1.0, 0.81 },
		INPUT{ 1.0, 1.0, 0.86 },
		INPUT{ 1.0, 1.0, 0.95 },
		INPUT{ 1.0, 0.0, 0.45 },
		INPUT{ 1.0, 1.0, 0.70 },
		INPUT{ 1.0, 0.0, 0.51 },
		INPUT{ 1.0, 1.0, 0.89 },
		INPUT{ 1.0, 1.0, 0.79 },
		INPUT{ 1.0, 0.0, 0.54 }
	};
	double** trainingOut = new double* [TRAINING_SETS] {
		OUTPUT{ 1.0, 0.0 },
		OUTPUT{ 1.0, 0.0 },
		OUTPUT{ 1.0, 0.0 },
		OUTPUT{ 1.0, 0.0 },
		OUTPUT{ 1.0, 0.0 },
		OUTPUT{ 0.0, 1.0 },
		OUTPUT{ 0.0, 1.0 },
		OUTPUT{ 0.0, 1.0 },
		OUTPUT{ 0.0, 1.0 },
		OUTPUT{ 0.0, 1.0 }
	};
	nn::BackpropagationTrainer trainer = nn::BackpropagationTrainer(0.1, 1e-4, 1000);

	trainNN(net, trainer, TRAINING_SETS, trainingIn, INPUTS, trainingOut, OUTPUTS);

	DELETE_TRAINING_DATA(trainingIn);
	DELETE_TRAINING_DATA(trainingOut);
}

int seed = 0;

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
	case '4': nnBackpropagation();
		break;
	case 'r':
		printf("Enter seed: ");

		string val;
		getline(cin, val);

		seed = stoi(val);
		srand(seed);

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
		printf("  4: Neural Network - Backpropagation\n");
		printf("  r: Reseed\n");
		printf("  q: quit\n");
		printf("? ");

		getline(cin, val);

		srand(seed);

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