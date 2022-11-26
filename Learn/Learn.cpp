/*
* This project is based off of
* Description Learner: Herbert Schildt - Chapter 7: "Machine Learning", Artificial Intelligence Using C
* Neural Network: Alan Souza, Fábio Soares - Neural Network Programming with Java (2016)
*/

#include <cmath>

#include <iostream>
#include <chrono>
#include "DescriptionLearner.h"
#include "ml/DLearnerListData.h"

//#define SPEEDTEST_MODE
//#define FAST_MODE

#include "NeuralNetwork.h"
#include "nn/PerceptronTrainer.h"
#include "nn/AdalineTrainer.h"
#include "nn/BackpropagationTrainer.h"
#include "nn/LevenbergMarquadtTrainer.h"
#include "nn/WTATrainer.h"
#include "nn/KohonenTrainer.h"

#include <RapidCSV/rapidcsv.h>
#include <unordered_set>

using namespace nn;

static int seed = 0;

void descriptionLearner() {
	ml::DescriptionLearner desc;
	desc.initialize(new ml::DLearnerListData());
	desc.execute();
}

struct TrainingData {
public:
	int TrainingSets;
	double** InputData;
	double** OutputData;
};

TrainingData getCSVTrainingData(const char* fname, int inputs, int outputs, bool splitOutput) {
	rapidcsv::Document doc(fname);

	std::vector<string> header = doc.GetRow<string>(0);
	int rows = doc.GetRowCount();
	int cols = header.size();

	int trainingSets = (rows - 1) / 20;

	int expOutputCols = splitOutput ? 1 : outputs;

	if(cols != inputs + expOutputCols)
		throw invalid_argument("Mismatch in expected input/output count vs number of columns in CSV.");

	TrainingData td{
		trainingSets,
		new double*[trainingSets],
		new double*[trainingSets]
	};


	std::minstd_rand eng(seed);
	std::uniform_int_distribution<> dist(0, trainingSets - 1);

	std::map<int, int> randIndices;
	std::unordered_set<int> usedIndices;
	for (int i = 0; i < trainingSets; i++) {
		int idx;
		do {
			idx = dist(eng);
		} while (usedIndices.count(idx) != 0);

		usedIndices.insert(idx);
		randIndices.emplace(i, idx);
	}

	int tSet = 0;
	for (int r = 1; r < rows, tSet < trainingSets; r+=20) {
		std::vector<double> row = doc.GetRow<double>(r);

		if (row.size() != cols)
			throw invalid_argument("Can only read rectangular CSV files.");

		double* inData = new double[inputs];
		double* outData = new double[outputs];

		int c = 0;
		for (int i = 0; c < inputs; i++) {
			inData[i] = row[c++];
		}

		if (splitOutput) {
			int value = doc.GetCell<int>(c, r);
			for (int o = 0; o < outputs; o++) {
				outData[o] = (o == value) ? 1 : 0;
			}
		}
		else {
			for (int o = 0; c < cols; o++) {
				outData[o] = row[c++];
			}
		}

		int tSetRand = randIndices[tSet];
		td.InputData[tSetRand] = inData;
		td.OutputData[tSetRand] = outData;
		tSet++;
	}

	return td;
}

#define TRAINING_IN(name) double** name = new double* [TRAINING_SETS]

#define INPUT new double[INPUTS]
#define OUTPUT new double[OUTPUTS]

#define DELETE_TRAINING_DATA(n) for(int _i = 0; _i < TRAINING_SETS; _i++) \
{ delete[] n[_i]; } delete[] n; n = nullptr;
#define DELETE_CSV_TRAINING_DATA(n) for(int _i = 0; _i < td.TrainingSets; _i++) \
{ delete[] n[_i]; } delete[] n; n = nullptr;
#define DELETE_VALIDATION_DATA(n) for(int _i = 0; _i < VALIDATION_SETS; _i++) \
{ delete[] n[_i]; } delete[] n; n = nullptr;

template<class T>
inline void trainNN_Supervised(NeuralNetwork& net, T trainer, 
	int TRAINING_SETS,
	double** trainingIn, int INPUTS, double** trainingOut, int OUTPUTS)
{
	printf("### TRAINING NETWORK ###\n---------------------------\n");

#ifdef SPEEDTEST_MODE
	NeuralNetwork* newNet;
	long long avgTime = 0;

	for (int i = 0; i < 100; i++) {
		newNet = new NeuralNetwork(net);

		auto start = chrono::high_resolution_clock::now();
		trainer.train(*newNet, TRAINING_SETS, trainingIn, INPUTS, trainingOut, OUTPUTS);
		auto stop = chrono::high_resolution_clock::now();

		auto time =
			chrono::duration_cast<chrono::microseconds>(stop - start).count();
		avgTime += time;
	}
	avgTime /= 100; 

	printf("\n### NETWORK AFTER TRAINING ###\n---------------------------\n");
	net.displayChange(*newNet);
	printf("Exec time: %lldus\n", avgTime);
#else
	NeuralNetwork newNet = NeuralNetwork(net);

	auto start = chrono::high_resolution_clock::now();
	trainer.train(newNet, TRAINING_SETS, trainingIn, INPUTS, trainingOut, OUTPUTS);
	auto stop = chrono::high_resolution_clock::now();

	printf("\n### NETWORK AFTER TRAINING ###\n---------------------------\n");
	net.displayChange(newNet);
	printf("Exec time: %lldus\n", 
		chrono::duration_cast<chrono::microseconds>(stop - start).count());
#endif
}

template<class T>
inline void trainNN_Unsupervised(NeuralNetwork& net, T trainer,
	int TRAINING_SETS, double** trainingIn, int INPUTS, int OUTPUTS,
	double** validation, int VALIDATION_SETS)
{
	printf("### TRAINING NETWORK ###\n---------------------------\n");

	NeuralNetwork newNet = NeuralNetwork(net);

	auto start = chrono::high_resolution_clock::now();
	trainer.train(newNet, TRAINING_SETS, trainingIn, INPUTS);
	auto stop = chrono::high_resolution_clock::now();

	printf("\n### NETWORK AFTER TRAINING ###\n---------------------------\n");
	net.displayChange(newNet);
	printf("Exec time: %lldus\n",
		chrono::duration_cast<chrono::microseconds>(stop - start).count());

	printf("\n### VERIFICATION ###\n---------------------------\n");
	for (int s = 0; s < VALIDATION_SETS; s++) {
		double* results = net.execute(validation[s], INPUTS);

		printf("\n%-7d | IN : [ ", s);
		for (int i = 0; i < INPUTS; i++) {
			printf("%.8f", validation[s][i]);

			if (i + 1 != INPUTS) {
				printf(", ");
			}
			else break;
		}
		printf(" ]");

		int cluster = 0;
		double maxRes = DBL_MIN;
		printf("\n%-7d | OUT: [ ", s);
		for (int o = 0; o < OUTPUTS; o++) {
			double res = results[o];

			printf("%.8f", res);

			if (res > maxRes) {
				cluster = o;
				maxRes = res;
			}

			if (o + 1 != OUTPUTS) {
				printf(", ");
			}
			else break;
		}
		printf(" ]");

		printf("\n%-7s | Cluster %d", "Results", cluster + 1);
	}
}

// Training a single-layer perceptron to emulate an AND operation.
// The training algorithm used simply adjusts weights in the direction of error.
void nnPerceptron() {
	auto net = NeuralNetwork::MakeNetwork({
		new FFNeuronLayer<ScalarFunc::Linear>(3, "in"),
		new FFNeuronLayer<ScalarFunc::Step>(1, false, false, "out")
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

	PerceptronTrainer trainer = PerceptronTrainer(0.1, 0e1, 100);

	trainNN_Supervised(net, trainer, TRAINING_SETS, trainingIn, INPUTS, trainingOut, OUTPUTS);

	DELETE_TRAINING_DATA(trainingIn);
	DELETE_TRAINING_DATA(trainingOut);
}

// Training a single-layer perceptron to solve a regression problem.
// The inputs are the amount of traffic on side roads, and the neural network
// is used to determine the resulting amount of traffic on the main avenue.
// The training algorithm used adjusts weights in the direction of error times 
// the derivative of the activation function.
void nnAdaline() {
	auto net = NeuralNetwork::MakeNetwork({
		new FFNeuronLayer<ScalarFunc::Linear>(4, "in"),
		new FFNeuronLayer<ScalarFunc::Linear>(1, false, false, "out")
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
		OUTPUT { 0.23 },
		OUTPUT { 0.45 },
		OUTPUT { 0.74 },
		OUTPUT { 0.63 },
		OUTPUT { 0.10 },
	};

	AdalineTrainer trainer = AdalineTrainer(0.25, 5e-4, 1000, 0.3);

	trainNN_Supervised(net, trainer, TRAINING_SETS, trainingIn, INPUTS, trainingOut, OUTPUTS);

	DELETE_TRAINING_DATA(trainingIn);
	DELETE_TRAINING_DATA(trainingOut);
}

// Training a multi-layer perceptron to solve a regression problem.
// The inputs are ...
// The training algorithm used adjusts weights ...
void nnBackpropagation() {
	auto net = NeuralNetwork::MakeNetwork({
		new FFNeuronLayer<ScalarFunc::Linear>(2, "in"),
		new FFNeuronLayer<ScalarFunc::LeakyReLU>(32, "hidden #1"),
		//new FFNeuronLayer<ScalarFunc::Siglog>(3, "out")
		new FFVNeuronLayer<VectorFunc::Softmax>(3, "out") 
	});

	for (int l = 0; l < net.depth(); l++) {
		NeuralNetwork::Layer& layer = net.getLayer(l);

		double stdev = sqrt(2.0 / (layer.size() * layer.inputsPerNeuron()));
		layer.initWeights<WeightInit::Normal, double, double, int>(stdev, 0, seed);
		//layer.initWeights<WeightInit::Uniform, double, double>(-1, 1);
	}

	constexpr int INPUTS = 2;
	constexpr int OUTPUTS = 3;

	TrainingData td = getCSVTrainingData("../files/spirals.csv", INPUTS, OUTPUTS, true);

	BackpropagationTrainer trainer = BackpropagationTrainer(0.05, 1e-2, 100, 0);

	trainNN_Supervised(net, trainer, td.TrainingSets, td.InputData, INPUTS, td.OutputData, OUTPUTS);

	DELETE_CSV_TRAINING_DATA(td.InputData);
	DELETE_CSV_TRAINING_DATA(td.OutputData);
}

// Training a multi-layer perceptron to solve a regression problem.
// The inputs are ...
// The training algorithm used adjusts weights ...
void nnLevenbergMarquadt() {
	auto net = NeuralNetwork::MakeNetwork({
		new FFNeuronLayer<ScalarFunc::Linear>(3, "in"),
		new FFNeuronLayer<ScalarFunc::Siglog>(3, "hidden"),
		new FFNeuronLayer<ScalarFunc::Linear>(2, "out")
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
	LevenbergMarquadtTrainer trainer = LevenbergMarquadtTrainer(0.1, 5e-2, 200);

	trainNN_Supervised(net, trainer, TRAINING_SETS, trainingIn, INPUTS, trainingOut, OUTPUTS);

	DELETE_TRAINING_DATA(trainingIn);
	DELETE_TRAINING_DATA(trainingOut);
}

// Training a self-organizing map to categorize inputs.
// The inputs are ...
// The training algorithm used adjusts weights ...
void nnWTA() {
	auto net = NeuralNetwork::MakeNetwork({
		new FFNeuronLayer<ScalarFunc::Linear>(3, "in"),
		new FFNeuronLayer<ScalarFunc::Linear>(2, "out")
	});

	constexpr int TRAINING_SETS = 6;
	constexpr int VALIDATION_SETS = 2;
	constexpr int INPUTS = 3;
	constexpr int OUTPUTS = 2;

	double** training = new double* [TRAINING_SETS] {
		INPUT{  1.0, -1.0,  1.0 },
		INPUT{ -1.0, -1.0, -1.0 },
		INPUT{ -1.0, -1.0,  1.0 },
		INPUT{  1.0,  1.0, -1.0 },
		INPUT{ -1.0,  1.0,  1.0 },
		INPUT{  1.0, -1.0, -1.0 }
	};
	double** validation = new double* [VALIDATION_SETS] {
		INPUT{ -1.0, 1.0, -1.0 },
		INPUT{  1.0, 1.0,  1.0 }
	};

	WTATrainer trainer = WTATrainer(0.1, 1e-4, 100);

	trainNN_Unsupervised(net, trainer, TRAINING_SETS, training,
		INPUTS, OUTPUTS, validation, VALIDATION_SETS);

	DELETE_TRAINING_DATA(training);
	DELETE_VALIDATION_DATA(validation);
}

// Training a self-organizing map to categorize inputs.
// The inputs are ...
// The training algorithm used adjusts weights ...
void nnKohonen() {
	auto net = NeuralNetwork::MakeNetwork({
		new FFNeuronLayer<ScalarFunc::Siglog>(3, "in"),
		new FFNeuronLayer<ScalarFunc::Siglog>(3, "hidden"),
		new FFNeuronLayer<ScalarFunc::Linear>(2, "out")
	});

	constexpr int TRAINING_SETS = 6;
	constexpr int VALIDATION_SETS = 2;
	constexpr int INPUTS = 3;
	constexpr int OUTPUTS = 2;

	double** training = new double* [TRAINING_SETS] {
		INPUT{  1.0, -1.0,  1.0 },
		INPUT{ -1.0, -1.0, -1.0 },
		INPUT{ -1.0, -1.0,  1.0 },
		INPUT{  1.0,  1.0, -1.0 },
		INPUT{ -1.0,  1.0,  1.0 },
		INPUT{  1.0, -1.0, -1.0 }
	};
	double** validation = new double* [VALIDATION_SETS] {
		INPUT{ -1.0, 1.0, -1.0 },
		INPUT{  1.0, 1.0,  1.0 }
	};

	KohonenTrainer trainer = KohonenTrainer(0.1, 1e-4, 100);

	trainNN_Unsupervised(net, trainer, TRAINING_SETS, training,
		INPUTS, OUTPUTS, validation, VALIDATION_SETS);

	DELETE_TRAINING_DATA(training);
	DELETE_VALIDATION_DATA(validation);
}

void execute(char ch) {
	if (ch == 'q') {
		exit(0);
	}
	else if (ch == '1') {
		descriptionLearner();
	}
	else if (ch == '2') {
		nnPerceptron();
	}
	else if (ch == '3') {
		nnAdaline();
	}
	else if (ch == '4') {
		nnBackpropagation();
	}
	else if (ch == '5') {
		nnLevenbergMarquadt();
	}
	else if (ch == '6') {
		nnWTA();
	}
	else if (ch == '7') {
		nnKohonen();
	}
	/*else if (ch == 'n') {
		printf("Enter training set: ");

		string val;
		getline(cin, val);

	}*/
	else if (ch == 'r') {
		printf("Enter seed: ");

		string val;
		getline(cin, val);

		seed = stoi(val);

	} // switch (ch)
}

int main()
{
	string val;
	for (;;) {
		// SLP/MLP - single/multilayer perceptron, aka fully connected feedforward neural networks.
		printf("Choose a program:\n");
		printf("  1: Description Learner\n");
		printf("  2: Neural Network - SLP, 'Perceptron Trainer' (gradient descent for simplest SLPs)\n");
		printf("  3: Neural Network - SLP, 'Adaline Trainer' (first-order gradient descent for SLPs)\n");
		printf("  4: Neural Network - MLP, 'Backpropagation Trainer' (first-order gradient descent)\n");
		printf("  5: Neural Network - MLP, 'Levenberg-Marquadt Trainer' (second-order gradient descent)\n");
		printf("  6: Neural Network - MLP/SOM, 'Winner Takes All Trainer'\n");
		printf("  7: Neural Network - MLP/SOM, 'Kohonen Trainer'\n");
		//printf("  n: Neural Network - Mix & Match\n");
		printf("  r: Reseed\n");
		printf("  q: quit\n");
		printf("? ");

		getline(cin, val);

		if (val.size() > 0) {
			char ch = val.c_str()[0];
#ifndef _DEBUG
			try {
#endif
				execute(ch);
#ifndef _DEBUG
			}
			catch (exception e) {
				printf("Failed: %s", e.what());
			}
#endif
		} // if 
		printf("\n\n");
	} // for
}