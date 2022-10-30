#pragma once

#include <list>
#include <map>
#include <string>
#include "DLearnerData.h"

namespace ml {
    /// <summary>
    /// Normal implementation of a database for the object description learner.
    /// </summary>
    class DLearnerListData : public DLearnerData {
    private:
        struct attribute {
            string subject;
            string verb;
            string object;
        };

        list<attribute> mayHave;
        list<attribute> mustHave;
        map<string, attribute> mayHaveMap;
        map<string, attribute> mustHaveMap;

        bool equals(attribute attr, string sub, string verb, string obj) {
            return attr.subject == sub && attr.verb == verb && attr.object == obj;
        }

    public:
        void initialize() {
        }

        void display() {
            list<attribute>::iterator it;

            printf("\nmay have:\n");
            for (it = mayHave.begin(); it != mayHave.end(); it++) {
                attribute val = *it;
                printf("  %s %s %s\n",
                    val.subject.c_str(), val.verb.c_str(), val.object.c_str());
            }

            printf("must have:\n");
            for (it = mustHave.begin(); it != mustHave.end(); it++) {
                attribute val = *it;
                printf("  %s %s %s\n",
                    val.subject.c_str(), val.verb.c_str(), val.object.c_str());
            }
        }

    public:
        void assertMust(string sub, string verb, string obj) {
            mustHave.push_back(attribute{ sub, verb, obj });
        }

        void assertMay(string sub, string verb, string obj) {
            mayHave.push_back(attribute{ sub, verb, obj });
        }

        bool retractMust(string sub, string verb, string obj) {
            list<attribute>::iterator it;
            for (it = mustHave.begin(); it != mustHave.end(); it++) {
                if (equals(*it, sub, verb, obj)) {
                    mustHave.erase(it);
                    return true;
                }
            }

            printf("failed to retract 'must': \"%s %s %s\"",
                sub.c_str(), verb.c_str(), obj.c_str());
            return false;
        }
        bool retractMay(string sub, string verb, string obj) {
            list<attribute>::iterator it;
            for (it = mayHave.begin(); it != mayHave.end(); it++) {
                if (equals(*it, sub, verb, obj)) {
                    mayHave.erase(it);
                    return true;
                }
            }

            printf("failed to retract 'may': \"%s %s %s\"",
                sub.c_str(), verb.c_str(), obj.c_str());
            return false;
        }

        bool findMay(string sub, string verb, string obj) {
            list<attribute>::iterator it;
            for (it = mayHave.begin(); it != mayHave.end(); it++) {
                attribute attr = *it;
                if (equals(attr, sub, verb, obj)) {
                    return true;
                }
            }
            return false;
        }

        bool generalizeMay(string sub, string verb, string obj) {
            list<attribute>::iterator it;
            for (it = mayHave.begin(); it != mayHave.end(); it++) {
                attribute* attr = &*it;

                // different subject
                if (attr->object == obj && attr->verb == verb
                    && attr->subject != sub) {
                    attr->subject = attr->subject + " or " + sub;
                }

                // different object
                if (attr->subject == sub && attr->verb == verb
                    && attr->object != obj) {
                    attr->object = attr->object + " or " + obj;
                }
            }

            for (it = mustHave.begin(); it != mustHave.end(); it++) {
                attribute* attr = &*it;

                // different subject
                if (attr->object == obj && attr->verb == verb
                    && attr->subject != sub) {
                    attr->subject = attr->subject + " or " + sub;

                    if (findMay(sub, verb, obj)) {
                        retractMay(sub, verb, obj);
                    }
                }

                // different object
                if (attr->subject == sub && attr->verb == verb
                    && attr->object != obj) {
                    attr->object = attr->object + " or " + obj;

                    if (findMay(sub, verb, obj)) {
                        retractMay(sub, verb, obj);
                    }
                }
            }

            return false;
        }
    };
}