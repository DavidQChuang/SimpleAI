/******************************************************************************

                              Online C++ Compiler.
               Code, Compile, Run and Debug C++ program online.
Write your code in this editor and press "Run" button to compile and execute it.

*******************************************************************************/

#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>

#include <string>

using namespace std;

enum class ScalarFunc {
    Step, Linear, Siglog, Hypertan, ReLU, LeakyReLU, GeLU,
};

class INeuronLayer {
public:
    virtual double func(double x) = 0;
};

template<ScalarFunc func>
class FFNeuronLayer : public INeuronLayer {};

template<>
class FFNeuronLayer<ScalarFunc::Step> : public INeuronLayer {
public:
    double func(double x) { return !signbit(x); }
};

template<>
class FFNeuronLayer<ScalarFunc::Linear> : public INeuronLayer {
public:
    double func(double x) { return x; }
};

template<typename... Args>
class NeuralNetwork {
private:
    typedef std::tuple<Args...> NNLayers;

    NNLayers nnLayers;

    template<std::size_t... Is>
    double execute(double x, std::index_sequence<Is...>) {
        double sum = x;
        auto sumElem = [&sum](auto& layer) {
            sum += layer.func(sum);
        };

        (sumElem(std::get<Is>(nnLayers)), ...);

        return sum;
    }

public:
    NeuralNetwork(Args... args) {
        nnLayers = std::tuple<Args...>(args...);
    }

    double execute(double x) {
        constexpr size_t size = std::tuple_size_v<NNLayers>;

        return execute(x, std::make_index_sequence<size>{});
    }
};


class NeuralNetwork2 {
private:
    std::vector<unique_ptr<INeuronLayer>> nnLayers;

public:
    NeuralNetwork2(std::initializer_list<INeuronLayer*> layers) {
        for (INeuronLayer* ptr : layers) {
            nnLayers.push_back(std::unique_ptr<INeuronLayer>(ptr));
        }
    }

    double execute(double x) {
        double sum = x;

        for (int i = 0; i < nnLayers.size(); i++) {
            sum += nnLayers[i]->func(sum);
        }

        return sum;
    }
};

void test1() {

    auto net2 = NeuralNetwork2({
        new FFNeuronLayer<ScalarFunc::Step>(),
        new FFNeuronLayer<ScalarFunc::Linear>(),
        });

    auto start = chrono::high_resolution_clock::now();

    double sum = 0;
    for (int i = 0; i < 1000000; i++) {
        sum += net2.execute(i / 10000.0);
    }

    cout << sum << endl;
    auto stop = chrono::high_resolution_clock::now();

    printf("Exec time: %lldus\n",
        chrono::duration_cast<chrono::microseconds>(stop - start).count());
}

void test2() {
    auto net = NeuralNetwork<
        FFNeuronLayer<ScalarFunc::Step>,
        FFNeuronLayer<ScalarFunc::Linear>
    >(
        FFNeuronLayer<ScalarFunc::Step>(),
        FFNeuronLayer<ScalarFunc::Linear>());

    auto start = chrono::high_resolution_clock::now();

    double sum = 0;
    for (int i = 0; i < 1000000; i++) {
        sum += net.execute(i / 10000.0);
    }

    cout << sum << endl;
    auto stop = chrono::high_resolution_clock::now();

    printf("Exec time: %lldus\n",
        chrono::duration_cast<chrono::microseconds>(stop - start).count());
}

int main()
{
    string val;
    for (;;) {
        test1();
        test2();

        std::getline(std::cin, val);

        if (val.size() > 0) {
            exit(0);
        } // if 
        printf("\n\n");
    } // for

    return 0;
}
