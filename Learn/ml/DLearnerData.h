#pragma once
#include <string>

using namespace std;

namespace ml {
    class DLearnerData {
    public:
        virtual void initialize() = 0;
        virtual void display() = 0;

        virtual void assertMust(string sub, string verb, string obj) = 0;
        virtual void assertMay(string sub, string verb, string obj) = 0;

        virtual bool retractMust(string sub, string verb, string obj) = 0;
        virtual bool retractMay(string sub, string verb, string obj) = 0;

        virtual bool findMay(string sub, string verb, string obj) = 0;

        virtual bool generalizeMay(string sub, string verb, string obj) = 0;

    };
}