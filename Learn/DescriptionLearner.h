#pragma once

#include <string>
#include <algorithm>
#include <iostream>

#include "ml/DLearnerData.h"

using namespace std;

namespace ml {
	/// <summary>
	/// Learns the descriptions of objects using simple machine learning.
	/// </summary>
	class DescriptionLearner {
	private:
		DLearnerData* data;

		void gets(string& str) {
			getline(cin, str, '\n');
		}

	public:
		void initialize(DLearnerData* data) {
			this->data = data;
		}

		void execute() {
			for (;;) {
				printf("(L)earn, (D)isplay, or (Q)uit? ");
				string ch;
				gets(ch);

				std::for_each(ch.begin(), ch.end(), [](char& c) {
					c = ::tolower(c);
					});

				switch (ch.c_str()[0]) {
				case 'l': learn();
					break;
				case 'd': display();
					break;
				case 'q': exit(0);
					break;
				}
				printf("\n");
			}
		}

	public:
		void learn() {
			string sub, verb, obj;
			string msub, mverb, mobj;

			for (;;) {
				printf("\nEnter an example.\n");

				// 
				if (!getExample(sub, verb, obj)) {
					return;
				}

				// If the statement hasn't been added yet, 
				// assert it and generalize statements matching its
				// signature.
				if (!data->findMay(sub, verb, obj)) {
					data->assertMay(sub, verb, obj);
					generalize(sub, verb, obj);
				}

				// Get an example of a near-miss statement
				// and restrict statements
				printf("\nEnter a near-miss (enter to skip):\n");
				getExample(msub, mverb, mobj);
				restrict(msub, mverb, mobj);
			}
		}

		void display() {
			data->display();
		}

	private:
		bool getExample(string& msub, string& mverb, string& mobj) {
			printf("subject: ");
			gets(msub);
			if (msub.empty()) return false;

			printf("verb: ");
			gets(mverb);
			if (mverb.empty()) return false;

			printf("object: ");
			gets(mobj);
			if (mobj.empty()) return false;

			return true;
		}

		// must be a statement of what something isn't
		void restrict(string ms, string mv, string mo){
			if (mv.size() <= 4 || mv.substr(0, 3) != "not") {
				printf("failed, too short or otherwise invalid.\n");
				return;
			}

			mv = mv.substr(4);
			if (data->findMay(ms, mv, mo)) {
				data->retractMay(ms, mv, mo);
				data->assertMust(ms, mv, mo);
			}
		}

			void generalize(string ms, string mv, string mo) {
			data->generalizeMay(ms, mv, mo);
		}
	};
}