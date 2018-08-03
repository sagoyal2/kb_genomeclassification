# -*- coding: utf-8 -*-
#BEGIN_HEADER
# The header block is where all import statments should live

# The header block is where all import statments should live
#from __future__ import division
#from __future__ import absolute_import
from __future__ import division

import os
import uuid
import codecs
#from Bio import SeqIO
#from pprint import pprint, pformat
#from AssemblyUtil.AssemblyUtilClient import AssemblyUtil
#from KBaseReport.KBaseReportClient import KBaseReport

    #here are some more imports for sklearn
#from sklearn import train_test_split
#from sklearn.grid_search import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import feature_selection
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold
#from sklearn.model_selection import StratifiedKFold
#from sklearn import StratifiedKFold
#from sklearn.grid_search import StratifiedKFold

#fix later
import seaborn as sns


import matplotlib.pyplot as plt
plt.switch_backend('agg')
#plt.switch_backend('TkAgg')


#%matplotlib inlines

    #here are imports for specific classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier

import numpy
import numpy as np
import pickle

import pandas as pd

from sklearn.tree import export_graphviz
import graphviz
import os
import re
import StringIO
import io
from io import open
import sys
from itertools import izip

#below are necessary for running in narrative
import time
import json
import os
import sys
import re
import requests
import numpy as np
import pprint
import scipy.spatial.distance as ssd
from matplotlib import pyplot as plt
#from scipy.cluster.hierarchy import dendrogram, linkage
#from biokbase.narrative.jobs.appmanager import AppManager
#from requests.packages.urllib3.exceptions import InsecurePlatformWarning
#requests.packages.urllib3.disable_warnings(InsecurePlatformWarning)
#from biokbase.narrative.jobs.appmanager import AppManager
import itertools

from biokbase.workspace.client import Workspace
from biokbase.workspace.client import Workspace as workspaceService

import pandas as pd

from KBaseReport.KBaseReportClient import KBaseReport
#from KBaseReportPy.KBaseReportPyClient import KBaseReportPy

from DataFileUtil.DataFileUtilClient import DataFileUtil

import xlrd


#END_HEADER


class kb_genomeclassification:
    '''
    Module Name:
    kb_genomeclassification

    Module Description:
    A KBase module: kb_genomeclassification
This module build a classifier and predict phenotypes based on the classifier
    '''

    ######## WARNING FOR GEVENT USERS ####### noqa
    # Since asynchronous IO can lead to methods - even the same method -
    # interrupting each other, you must be *very* careful when using global
    # state. A method could easily clobber the state set by another while
    # the latter method is running.
    ######################################### noqa
    VERSION = "0.0.1"
    GIT_URL = "https://github.com/janakagithub/kb_genomeclassification.git"
    GIT_COMMIT_HASH = "511b3b0cbc447d38ea281a52806194aed7410016"

    #BEGIN_CLASS_HEADER
    # Class variables and functions can be defined in this block

    def classifierTest(self, classifier, classifier_name, my_mapping, master_Role,  splits, train_index, test_index, print_cfm):
        """
        args:
        ---classifier which is a sklearn object that has methods #LogisticRegression()
        ---classifier_name is a string and is what is the name given to what the classifer is being saved as
        ---print_cfm is boolean (False when running through tuning and you don't want to print out all results on the
                                console - True otherwise) you might need to rethink this value when implementing and saving classifiers

        does:
        ---calculates the numerical value of the the classifiers
        ---saves down pickled versions of classifiers (probably make a separate method)
            ---saves down base64 versions of classifiers
        ---creates the text fields and values to be placed into the statistics table
        ---calls the plot_confusion_matrix function

        return:
        --- (numpy.average(train_score), numpy.std(train_score), numpy.average(validate_score), numpy.std(validate_score))
            ---return statement is only used when you repeatedly loop through this function during tuning
        """
        if print_cfm:
            print classifier_name
            self.list_name.extend([classifier_name])
        train_score = numpy.zeros(splits)
        validate_score = numpy.zeros(splits)
        matrix_size = self.class_list.__len__()

        cnf_matrix = numpy.zeros(shape=(matrix_size, matrix_size))
        cnf_matrix_f = numpy.zeros(shape=(matrix_size, matrix_size))
        for c in xrange(splits):
            X_train = self.full_attribute_array[train_index[c]]
            y_train = self.full_classification_array[train_index[c]]
            X_test = self.full_attribute_array[test_index[c]]
            y_test = self.full_classification_array[test_index[c]]
            classifier.fit(X_train, y_train)
            train_score[c] = classifier.score(X_train, y_train)
            validate_score[c] = classifier.score(X_test, y_test)
            y_pred = classifier.predict(X_test)
            cnf = confusion_matrix(y_test, y_pred)
            cnf_f = cnf.astype(u'float') / cnf.sum(axis=1)[:, numpy.newaxis]
            for i in xrange(len(cnf)):
                for j in xrange(len(cnf)):
                    cnf_matrix[i][j] += cnf[i][j]
                    cnf_matrix_f[i][j] += cnf_f[i][j]

        if print_cfm:
            pickle_out = open(u"/kb/module/work/tmp/forDATA/" + unicode(classifier_name) + u".pickle", u"wb")

            #pickle_out = open("/kb/module/work/tmp/" + str(self.classifier_name) + ".pickle", "wb")


            pickle.dump(classifier.fit(self.full_attribute_array, self.full_classification_array), pickle_out, protocol = 2)
            pickle_out.close()


            #current_pickle = pickle.dumps(classifier.fit(self.full_attribute_array, self.full_classification_array), protocol=0)
            #pickled = codecs.encode(current_pickle, "base64").decode()


            """

            with open(u"/kb/module/work/tmp/" + unicode(classifier_name) + u".txt", u"w") as f:
                for line in pickled:
                    f.write(line)
            """

        pickled = "this is what the pickled string would be"

        print ""
        print "This is printing out the classifier_object that needs to be saved down dump"
        print ""

        classifier_object= {
        'classifier_id' : '',
        'classifier_type' : classifier_name, # Neural network
        'classifier_name' : classifier_name,
        'classifier_data' : pickled,
        'classifier_description' : 'this is my description',
        'lib_name' : 'sklearn',
        'attribute_type' : 'functional_roles',
        'number_of_attributes' : self.class_list.__len__(),
        'attribute_data' : master_Role, #master_Role,
        'class_list_mapping' : my_mapping, #my_mapping,
        'number_of_genomes' : 0,
        'training_set_ref' : ''
        }

        print classifier_object

        list_forDict = []

        if self.class_list.__len__() == 3:
            if print_cfm:
                cnf_av = cnf_matrix / splits
                print
                print cnf_av[0][0], cnf_av[0][1], cnf_av[0][2]
                print cnf_av[1][0], cnf_av[1][1], cnf_av[1][2]
                print cnf_av[2][0], cnf_av[2][1], cnf_av[2][2]
                print
                print self.class_list[0]
                TP = cnf_av[0][0]
                TN = cnf_av[1][2] + cnf_av[1][2] + cnf_av[2][1] + cnf_av[2][2]
                FP = cnf_av[0][1] + cnf_av[0][2]
                FN = cnf_av[1][0] + cnf_av[2][0]
                list_forDict.extend([None])
                list_forDict.extend(self.cf_stats(TN, TP, FP, FN))

                print self.class_list[1]
                TP = cnf_av[1][1]
                TN = cnf_av[0][0] + cnf_av[0][2] + cnf_av[2][0] + cnf_av[2][2]
                FP = cnf_av[1][0] + cnf_av[1][2]
                FN = cnf_av[0][1] + cnf_av[2][1]
                list_forDict.extend([None, None])
                list_forDict.extend(self.cf_stats(TN, TP, FP, FN))

                print self.class_list[2]
                TP = cnf_av[2][2]
                TN = cnf_av[0][0] + cnf_av[0][1] + cnf_av[1][0] + cnf_av[1][1]
                FP = cnf_av[2][0] + cnf_av[2][1]
                FN = cnf_av[0][1] + cnf_av[0][2]
                list_forDict.extend([None, None])
                list_forDict.extend(self.cf_stats(TN, TP, FP, FN))

                list_forDict.extend([(list_forDict[4] + list_forDict[10] + list_forDict[16])/3])

                self.list_statistics.append(list_forDict)

                # self.plot_confusion_matrix(cnf_matrix/10,class_list,'Confusion Matrix')
                self.plot_confusion_matrix(cnf_matrix_f/splits*100.0,self.class_list,u'Confusion Matrix',classifier_name)

        if self.class_list.__len__() == 2:
            if print_cfm:

                TP = cnf[0][0]
                TN = cnf[1][1]
                FP = cnf[0][1]
                FN = cnf[1][0]

                list_forDict.extend(self.cf_stats(TN, TP, FP, FN))
                self.list_statistics.append(list_forDict)

                self.plot_confusion_matrix(cnf_matrix_f/splits*100.0,self.class_list,u'Confusion Matrix',classifier_name)

        if print_cfm:
            print classifier
            print
            print u"Confusion matrix"
            for i in xrange(len(cnf_matrix)):
                print self.class_list[i],; sys.stdout.write(u"  \t")
                for j in xrange(len(cnf_matrix[i])):
                    print cnf_matrix[i][j] / splits,; sys.stdout.write(u"\t")
                print
            print
            for i in xrange(len(cnf_matrix_f)):
                print self.class_list[i],; sys.stdout.write(u"  \t")
                for j in xrange(len(cnf_matrix_f[i])):
                    print u"%6.1f" % ((cnf_matrix_f[i][j] / splits) * 100.0),; sys.stdout.write(u"\t")
                print
            print
            print u"01", cnf_matrix[0][1]

        print u"%6.3f\t%6.3f\t%6.3f\t%6.3f" % (
        numpy.average(train_score), numpy.std(train_score), numpy.average(validate_score), numpy.std(validate_score))

        return (numpy.average(train_score), numpy.std(train_score), numpy.average(validate_score), numpy.std(validate_score))


    def whichClassifier(self, name):
        """
        args:
        ---name which is a string that the user will pass in as to which classifier (sklearn) classifier they want
        does:
        ---matches string with sklearn classifier
        return:
        ---sklearn classifier
        """

        if name == u"KNeighborsClassifier":
            return KNeighborsClassifier()
        elif name == u"GaussianNB":
            return GaussianNB()
        elif name == u"LogisticRegression":
            return LogisticRegression(random_state=0)
        elif name == u"DecisionTreeClassifier":
            return DecisionTreeClassifier(random_state=0)
        elif name == u"SVM":
            return svm.LinearSVC(random_state=0)
        elif name == u"NeuralNetwork":
            return MLPClassifier(random_state=0)
        else:
            return u"ERROR THIS SHOULD NOT HAVE REACHED HERE"

    def cf_stats(self, TN, TP, FP, FN):
        """
        args:
        ---TN int for True Negative
        ---TP int for True Positive
        ---FP int for False Positive
        ---FN int for False Negative
        does:
        ---calculates statistics as a way to measure and evaluate the performance of the classifiers
        return:
        ---list_return=[((TP + TN) / Total), (Precision), (Recall), (2 * ((Precision * Recall) / (Precision + Recall)))]
            ---((TP + TN) / Total)) == Accuracy
            --- Precision
            --- Recall
            ---(2 * ((Precision * Recall) / (Precision + Recall))) == F1 Score

        ---
        """

        AN = TN + FP
        AP = TN + FN
        PN = TN + FN
        PP = TP + FP
        Total = TN + TP + FP + FN
        Recall = (TP / (TP + FN))
        Precision = (TP / (TP + FP))
        """
        print("Accuracy:\t\t%6.3f" % ((TP + TN) / Total))
        print("Precision:\t\t%6.3f" % (Precision))
        print("Recall:\t\t%6.3f" % (Recall))
        print("F1 score::\t\t%6.3f" % (2 * ((Precision * Recall) / (Precision + Recall))))
        print()
        """

        list_return=[((TP + TN) / Total), (Precision), (Recall), (2 * ((Precision * Recall) / (Precision + Recall)))]



        return list_return

    def to_HTML_Statistics(self, additional = False):
        """
        args:
        ---additional is a boolean and is used to indicate if this method is being called to make html2
        does:
        ---the statistics that were calculated and stored into lists are converted into a dataframe table --> html page
        return:
        ---N/A but instead creates an html file in tmp
        """


        #self.counter = self.counter + 1

        if not additional:

            print u"I am inside not additional"

            statistics_dict = {}

            #print(self.list_name)
            #print(self.list_statistics)

            for i, j in izip(self.list_name, self.list_statistics):
                statistics_dict[i] = j

            data = statistics_dict

            if self.class_list.__len__() == 3:
                my_index = [u'Facultative', u'Accuracy:', u'Precision:', u'Recall:', u'F1 score::', None, u'Aerobic', u'Accuracy:',
                        u'Precision:', u'Recall:', u'F1 score::', None, u'Anaerobic', u'Accuracy:', u'Precision:', u'Recall:',
                        u'F1 score::', u'Average F1']

            if self.class_list.__len__() == 2:
                my_index = [u'Accuracy:', u'Precision:', u'Recall:', u'F1 score::']

            df = pd.DataFrame(data, index=my_index)

            df.to_html(u'/kb/module/work/tmp/forHTML/newStatistics.html')

            df['Max'] = df.idxmax(1)
            best_classifier_str = df['Max'].iloc[-1]


            file = open(u'/kb/module/work/tmp/forHTML/newStatistics.html', u'r')
            allHTML = file.read()
            file.close()

            new_allHTML = re.sub(r'NaN', r'', allHTML)

            file = open(u'/kb/module/work/tmp/forHTML/newStatistics.html', u'w')
            file.write(new_allHTML)
            file.close

            return best_classifier_str

        if additional:
            statistics_dict = {}

            neededIndex = [2, 3, self.list_name.__len__() - 2, self.list_name.__len__() -1]
            #neededIndex = [self.list_name.__len__() - 2, self.list_name.__len__() -1]
            sub_list_name = [self.list_name[i] for i in neededIndex]
            sub_list_statistics = [self.list_statistics[i] for i in neededIndex]

            for i, j in izip(sub_list_name, sub_list_statistics):
                statistics_dict[i] = j

            data = statistics_dict

            if self.class_list.__len__() == 3:
                my_index = [u'Facultative', u'Accuracy:', u'Precision:', u'Recall:', u'F1 score::', None, u'Aerobic', u'Accuracy:',
                        u'Precision:', u'Recall:', u'F1 score::', None, u'Anaerobic', u'Accuracy:', u'Precision:', u'Recall:',
                        u'F1 score::', u'Average F1']

            if self.class_list.__len__() == 2:
                my_index = [u'Accuracy:', u'Precision:', u'Recall:', u'F1 score::']

            df = pd.DataFrame(data, index=my_index)
            df.to_html(u'/kb/module/work/tmp/forHTML/postStatistics.html')

            df['Max'] = df.idxmax(1)
            best_classifier_str = df['Max'].iloc[-1]

            file = open(u'/kb/module/work/tmp/forHTML/postStatistics.html', u'r')
            allHTML = file.read()
            file.close()

            new_allHTML = re.sub(r'NaN', r'', allHTML)

            file = open(u'/kb/module/work/tmp/forHTML/postStatistics.html', u'w')
            file.write(new_allHTML)
            file.close

            return best_classifier_str


        #df.to_html('statistics' + str(self.counter) + '.html')

    def plot_confusion_matrix(self,cm, classes, title,classifier_name):
        """
        args:
        ---cm is the "cnf_matrix" which is a numpy array of numerical values for the confusion matrix
        ---classes is the class_list which is a list of the classes ie. [N,P] or [Aerobic, Anaerobic, Facultative]
        ---title is a "heading" that appears on the image
        ---classifier_name is the classifier name and is what the saved .png file name will be
        does:
        ---creates a confusion matrix .png file and saves it
        return:
        ---N/A but instead creates an .png file in tmp

        """
        """
        plt.rcParams.update({'font.size': 18})
        #fig,ax= plt.subplots(figsize=(5,4))
        fig = plt.figure()
        ax = fig.add_subplot(figsize=(5,4))
        sns.set(font_scale=1.5)
        sns_plot = sns.heatmap(cm, annot=True, ax = ax, cmap="Blues"); #annot=True to annotate cells
        # labels, title and ticks
        ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels');
        ax.set_title(title);
        ax.xaxis.set_ticklabels(classes); ax.yaxis.set_ticklabels(classes);
        #fig.savefig(classifier_name+".png") #this may not be necessary as not necessary to save png file
        sns_plot.savefig(classifier_name+".png", format='png')
        """

        plt.rcParams.update({u'font.size': 18})
        fig = plt.figure()
        ax = fig.add_subplot(figsize=(5,5))
        sns.set(font_scale=1.5)
        sns_plot = sns.heatmap(cm, annot=True, ax = ax, cmap=u"Blues"); #annot=True to annotate cells
        ax = sns_plot
        ax.set_xlabel(u'Predicted labels'); ax.set_ylabel(u'True labels');
        ax.set_title(title);
        ax.xaxis.set_ticklabels(classes); ax.yaxis.set_ticklabels(classes);
        #ax.savefig(classifier_name+".png", format='png')

        fig = sns_plot.get_figure()
        #fig.savefig(u"./pics/" + classifier_name +u".png", format=u'png')
        fig.savefig(u"/kb/module/work/tmp/forHTML/" + classifier_name +u".png", format=u'png')

    def tree_code(self, optimized_tree, spacer_base=u"    "):
        """
        args:
        ---optimized_tree this is a DecisionTree object that has been tuned
        ---spacer_base is string physically acting as a spacer
        does:
        ---Produce psuedo-code for decision tree - based on http://stackoverflow.com/a/30104792.
        ---calls printTree
        return:
        ---N/A but prints out a visual of what the DecisionTree object looks like on the inside
        """

        tree = optimized_tree #DecisionTreeClassifier(random_state=0, max_depth=3, criterion='entropy')
        #tree = DecisionTreeClassifier(random_state=0, max_depth=3, criterion='entropy')
        print u"Hello"

        tree.fit(self.full_attribute_array, self.full_classification_array)

        feature_names = self.attribute_list
        target_names = self.class_list

        left = tree.tree_.children_left
        right = tree.tree_.children_right
        threshold = tree.tree_.threshold
        features = [feature_names[i] for i in tree.tree_.feature]
        value = tree.tree_.value

        def recurse(left, right, threshold, features, node, depth):
            spacer = spacer_base * depth
            if (threshold[node] != -2):
                print spacer + u"if ( " + features[node] + u" <= " + \
                      unicode(threshold[node]) + u" ) {"
                if left[node] != -1:
                    recurse(left, right, threshold, features,
                                 left[node], depth + 1)
                print spacer + u"}\n" + spacer + u"else {"
                if right[node] != -1:
                    recurse(left, right, threshold, features,
                                 right[node], depth + 1)
                print spacer + u"}"
            else:
                target = value[node]
                for i, v in izip(numpy.nonzero(target)[1],
                                target[numpy.nonzero(target)]):
                    target_name = target_names[i]
                    target_count = int(v)
                    print spacer + u"return " + unicode(target_name) + \
                          u" ( " + unicode(target_count) + u" examples )"

        recurse(left, right, threshold, features, 0, 0)

        self.printTree(tree, u"NAMEmyTreeLATER")

    def tune_Decision_Tree(self, splits, train_index, test_index):
        """
        args:
        ---NA
        does:
        ---by looping through various parameters (1. depth 2. criterion) it selects best configuration
        ---calls tree_code
        return:
        ---N/A but main function is just to figure out "workhorse"
        """

        """
        skf = StratifiedKFold(n_splits=splits, random_state=0, shuffle=True)
        for train_idx, test_idx in skf.split(self.full_attribute_array, self.full_classification_array):
            self.train_index.append(train_idx)
            self.test_index.append(test_idx)
        """

        #below code is for gini-criterion
        val = numpy.zeros(12)
        test_av = numpy.zeros(12)
        test_std = numpy.zeros(12)
        val_av = numpy.zeros(12)
        val_std = numpy.zeros(12)
        for d in xrange(1, 12):
            val[d] = d
            (test_av[d], test_std[d], val_av[d], val_std[d]) = self.classifierTest(DecisionTreeClassifier(random_state=0, max_depth=d), u"DecisionTreeClassifier", my_mapping, master_Role, splits, train_index, test_index, False)

        fig, ax = plt.subplots(figsize=(6, 6))
        plt.errorbar(val[1:], test_av[1:], yerr=test_std[1:], fmt=u'o', label=u'Training set')
        plt.errorbar(val[1:], val_av[1:], yerr=val_std[1:], fmt=u'o', label=u'Testing set')
        ax.set_ylim(ymin=0.5, ymax=1.1)
        ax.set_title(u"Gini Criterion")
        plt.xlabel(u'Tree depth', fontsize=12)
        plt.ylabel(u'Accuracy', fontsize=12)
        plt.legend(loc=u'lower left')
        #plt.savefig(u"./pics/"+ self.global_target +u"_gini_depth-met.png")
        #fig.savefig(u"/kb/module/work/tmp/pics/" + classifier_name +u".png", format=u'png')
        plt.savefig(u"/kb/module/work/tmp/forHTML/"+ self.global_target +u"_gini_depth-met.png")

        gini_best_index = np.argmax(val_av)
        print gini_best_index
        gini_best = np.amax(val_av)

        #below code is for entropy-criterion
        val = numpy.zeros(12)
        test_av = numpy.zeros(12)
        test_std = numpy.zeros(12)
        val_av = numpy.zeros(12)
        val_std = numpy.zeros(12)
        for d in xrange(1, 12):
            val[d] = d
            (test_av[d], test_std[d], val_av[d], val_std[d]) = self.classifierTest(DecisionTreeClassifier(random_state=0, max_depth=d, criterion=u'entropy'), u"DecisionTreeClassifier", my_mapping, master_Role, splits, train_index, test_index, False)

        fig, ax = plt.subplots(figsize=(6, 6))
        plt.errorbar(val[1:], test_av[1:], yerr=test_std[1:], fmt=u'o', label=u'Training set')
        plt.errorbar(val[1:], val_av[1:], yerr=val_std[1:], fmt=u'o', label=u'Testing set')
        ax.set_ylim(ymin=0.5, ymax=1.1)
        ax.set_title(u"Entropy Criterion")
        plt.xlabel(u'Tree depth', fontsize=12)
        plt.ylabel(u'Accuracy', fontsize=12)
        plt.legend(loc=u'lower left')
        #plt.savefig(u"./pics/"+ self.global_target +u"_entropy_depth-met.png")
        plt.savefig(u"/kb/module/work/tmp/forHTML/"+ self.global_target +u"_entropy_depth-met.png")

        entropy_best_index = np.argmax(val_av)
        print entropy_best_index
        entropy_best = np.amax(val_av)


        #gini_best_index = 4
        #entropy_best_index = 3

        self.classifierTest(DecisionTreeClassifier(random_state=0, max_depth=gini_best_index, criterion=u'gini'), self.global_target + u"_DecisionTreeClassifier(gini)", my_mapping, master_Role, splits, train_index, test_index,True)
        self.classifierTest(DecisionTreeClassifier(random_state=0, max_depth=entropy_best_index, criterion=u'entropy'), self.global_target + u"_DecisionTreeClassifier(entropy)", my_mapping, master_Role, splits, train_index, test_index,True)

        self.to_HTML_Statistics(additional=True)

        if gini_best > entropy_best:
            self.tree_code(DecisionTreeClassifier(random_state=0, max_depth=gini_best_index, criterion=u'gini'))
        else:
            self.tree_code(DecisionTreeClassifier(random_state=0, max_depth=entropy_best_index, criterion=u'entropy'))


    def parse_lookNice(self,name):
        """
        args:
        ---name is a string that is what you want the DecisionTree image saved as
        does:
        ---this cleans up the dot file to produce a more visually appealing tree figure using graphviz
        return:
        ---N/A but saves a .png of the name in the tmp folder
        """

        import re

        f = open(u"/kb/module/work/tmp/dotFolder/mydotTree.dot", u"r")
        allStr = f.read()
        f.close()
        new_allStr = allStr.replace(u'\\n', u'')

        first_fix = re.sub(ur'(\w\s\[label="[\w\s.,:()-]+)<=([\w\s.\[\]=,]+)("] ;)',
                           ur'\1 (Absent)" , color="0.650 0.200 1.000"] ;', new_allStr)
        second_fix = re.sub(ur'(\w\s\[label=")(.+?class\s=\s)', ur'\1', first_fix)

        # nominal fixes like color and shape
        third_fix = re.sub(ur'shape=box] ;', ur'shape=Mrecord] ; node [style=filled];', second_fix)

        if self.class_list.__len__() == 3:
            fourth_fix = re.sub(ur'(\w\s\[label="anaerobic")', ur'\1, color = "0.5176 0.2314 0.9020"', third_fix)
            fifth_fix = re.sub(ur'(\w\s\[label="aerobic")', ur'\1, color = "0.5725 0.6118 1.0000"', fourth_fix)
            sixth_fix = re.sub(ur'(\w\s\[label="facultative")', ur'\1, color = "0.5804 0.8824 0.8039"', fifth_fix)
            f = open(u"/kb/module/work/tmp/dotFolder/niceTree.dot", u"w")
            f.write(sixth_fix)
            f.close()

            os.system(u'dot -Tpng /kb/module/work/tmp/dotFolder/niceTree.dot >  '+ u"/kb/module/work/tmp/forHTML/"  + name + u'.png ')

        if self.class_list.__len__() == 2:
            fourth_fix = re.sub(ur'(\w\s\[label="N")', ur'\1, color = "0.5176 0.2314 0.9020"', third_fix)
            fifth_fix = re.sub(ur'(\w\s\[label="P")', ur'\1, color = "0.5725 0.6118 1.0000"', fourth_fix)
            f = open(u"/kb/module/work/tmp/dotFolder/niceTree.dot", u"w")
            f.write(fifth_fix)
            f.close()

            os.system(u'dot -Tpng /kb/module/work/tmp/dotFolder/niceTree.dot >  '+ u"/kb/module/work/tmp/forHTML/" + name + u'.png ')

    def html_report_1(self, classifier, best_classifier_str):
        """
        does: creates an .html file that makes the frist report (first app).
        """
        file = open(u"/kb/module/work/tmp/forHTML/html1.html", u"w")

        html_string = u"""
        <!DOCTYPE html>
        <html>
        <head>
        <style>
        table, th, td {
            border: 1px solid black;
            border-collapse: collapse;
        }

        * {
            box-sizing: border-box;
        }

        .column {
            float: left;
            width: 50%;
            padding: 10px;
        }

        /* Clearfix (clear floats) */
        .row::after {
            content: "";
            clear: both;
            display: table;
        }
        </style>
        </head>
        <body>

        <h1 style="text-align:center;">Metabolic Respiration Classifier</h1>

        <p style="text-align:center; font-size:160%;">  Prediction of respiration type based on classifiers depicted in the form of confusion matrices.  A.) K-Nearest-Neighbors Classifier B. ) Naive Gaussian Bayes Classifier C.) , Logistic Regression Classifier and the D.) Decision Tree Classifier E.) Linear SVM, F.) Neural Network</p>
        <h2> Disclaimer:No feature selection and parameter optimization was not done</h2>
        """

        file.write(html_string)

        if classifier == u"Default":
            next_str = u"""
        <div class="row">
          <div class="column">
              <p style="text-align:left; font-size:160%;">K-Nearest-Neighbors Classifier</p>
            <img src=" """+self.global_target +"""_KNeighborsClassifier.png" alt="Snow" style="width:100%">
              <!-- <figcaption>Fig.1 - Trulli, Puglia, Italy.</figcaption> -->
          </div>
          <div class="column">
              <p style="text-align:left; font-size:160%;">Logistic Regression Classifier</p>
            <img src=" """+ self.global_target +"""_LogisticRegression.png" alt="Snow" style="width:100%">
          </div>
        </div>

        <div class="row">
          <div class="column">
              <p style="text-align:left; font-size:160%;">Naive Gaussian Bayes Classifier</p>
            <img src=" """+ self.global_target +"""_GaussianNB.png" alt="Snow" style="width:100%">
          </div>
          <div class="column">
              <p style="text-align:left; font-size:160%;">Linear SVM Classifier</p>
            <img src=" """+ self.global_target +"""_SVM.png" alt="Snow" style="width:100%">
          </div>
        </div>

        <div class="row">
          <div class="column">
              <p style="text-align:left; font-size:160%;">Decision Tree Classifier</p>
            <img src=" """+ self.global_target +"""_DecisionTreeClassifier.png" alt="Snow" style="width:100%">
          </div>
          <div class="column">
              <p style="text-align:left; font-size:160%;">Neural Network Classifier</p>
            <img src=" """+ self.global_target +"""_NeuralNetwork.png" alt="Snow" style="width:100%">
          </div>
        </div>
            """
            file.write(next_str)

            next_str = u"""
            <p style="font-size:160%;">Comparison of statistics in the form of Accuracy, Precision, Recall and F1 Score calculated against the confusion matrices of respiration type for the classifiers</p>
            """
            file.write(next_str)

            another_file = open(u"/kb/module/work/tmp/forHTML/newStatistics.html", u"r")
            all_str = another_file.read()
            another_file.close()

            file.write(all_str)

        else:
            next_str = u"""
            <div class="row">
          <div class="column">
            <p style="text-align:left; font-size:160%;">""" + classifier + """</p>
            <img src=" """+ self.global_target +"""_""" + classifier + """.png" alt="Snow" style="width:100%">
          </div>
          <div class="column">
            """
            file.write(next_str)

            next_str = u"""
            <p style="font-size:160%;">Comparison of statistics in the form of Accuracy, Precision, Recall and F1 Score calculated against the confusion matrices of respiration type for the classifiers</p>
            """
            file.write(next_str)

            another_file = open(u"/kb/module/work/tmp/forHTML/newStatistics.html", u"r")
            all_str = another_file.read()
            another_file.close()

            file.write(all_str)

            next_str = u"""
            </div>
            </div>
            """
            file.write(next_str)


        next_str = u"""
         <p style="text-align:center; font-size:100%;">  Based on these results it would be in your best interest to use the """ + unicode(best_classifier_str) + """ as your model as
         it produced the strongest F1 score </p>
        """

        file.write(next_str)

        next_str = u"""
        <a href="html2.html">Second html page</a>
        """

        file.write(next_str)

        file.close()

    def html_report_2(self):
        """
        does: creates an .html file that makes the second report (first app).
        """
        file = open(u"/kb/module/work/tmp/forHTML/html2.html", u"w")

        html_string = u"""
        <!DOCTYPE html>
        <html>
        <head>
        <style>
        table, th, td {
            border: 1px solid black;
            border-collapse: collapse;
        }

        * {
            box-sizing: border-box;
        }

        .column {
            float: left;
            width: 50%;
            padding: 10px;
        }

        /* Clearfix (clear floats) */
        .row::after {
            content: "";
            clear: both;
            display: table;
        }
        </style>
        </head>
        <body>

        <h1 style="text-align:center;">Metabolic Respiration Classifier - Decision Tree Tuning</h1>

        <!-- <h2>Maybe we can add some more text here later?</h2> -->
        <!--<p>How to create side-by-side images with the CSS float property:</p> -->

        <p style="text-align:center; font-size:160%;">  Comparison of level of Accuracy between respiration Training versus Testing data sets based on  the Gini Criterion and the Entropy Criterion for 11 levels of Tree Depth </p>
        <p style="text-align:center; font-size:100%;">  (Below is the training and test accuracy at each tree depth. The Decision Criterion was Gini and Entropy) </p>
        """

        file.write(html_string)

        next_str = u"""

        <div class="row">
          <div class="column">
              <p style="text-align:left; font-size:160%;">Training vs Testing Score on Gini Criterion </p>
            <img src=" """+ self.global_target +"""_gini_depth-met.png" alt="Snow" style="width:100%">
              <!-- <figcaption>Fig.1 - Trulli, Puglia, Italy.</figcaption> -->
          </div>
          <div class="column">
              <p style="text-align:left; font-size:160%;">Training vs Testing Score on Entropy Criterion</p>
            <img src=" """+ self.global_target +"""_entropy_depth-met.png" alt="Snow" style="width:100%">
          </div>
        </div>

        <p style="text-align:center; font-size:160%;">  Comparison of respiration tuned Gini and Entropy based Decision Tree Classifiers depicted in the form of confusion matrices. A.) Decision Tree Classifier B.) Decision Tree Classifier-Gini C.) Decision Tree Classifier-Entropy D.) Naive Gaussian Bayes Classifier </p>
        <p style="text-align:center; font-size:100%;">  The original Decision Tree Classifier model was chosen as a base comparision and Logistic Regression model was chosen since it showed the best average F1 Score </p>

        <div class="row">
          <div class="column">
              <p style="text-align:left; font-size:160%;"> Decision Tree Classifier </p>
            <img src=" """+ self.global_target +"""_DecisionTreeClassifier.png" alt="Snow" style="width:100%">
          </div>
          <div class="column">
              <p style="text-align:left; font-size:160%;"> Logistic Regression Classifier </p>
            <img src=" """+ self.global_target +"""_LogisticRegression.png" alt="Snow" style="width:100%">
          </div>
        </div>

        <div class="row">
          <div class="column">
              <p style="text-align:left; font-size:160%;"> Decision Tree Classifier - Gini </p>
            <img src=" """+ self.global_target +"""_DecisionTreeClassifier(gini).png" alt="Snow" style="width:100%">
          </div>
          <div class="column">
              <p style="text-align:left; font-size:160%;"> Decision Tree Classifier - Entropy </p>
            <img src=" """+ self.global_target +"""_DecisionTreeClassifier(entropy).png" alt="Snow" style="width:100%">
          </div>
        </div>
        """
        file.write(next_str)

        next_str= u"""
        <p style="font-size:160%;">Comparison of statistics in the form of Accuracy, Precision, Recall and F1 Score calculated against the confusion matrices of respiration type for the classifiers</p>
        """
        file.write(next_str)

        another_file = open(u"/kb/module/work/tmp/forHTML/postStatistics.html", u"r")
        all_str = another_file.read()
        another_file.close()

        file.write(all_str)

        next_str= u"""
        <p style="font-size:160%;"> Below is a tree created that displays a visual for how genomes were classified.</p>
        <p style="font-size:100%;"> READ: if __functional__role__ is absent (true) then move left otherwise if __functional__role__ is present (false) move right</p>

        <img src="NAMEmyTreeLATER.png" alt="Snow" style="width:100%">

        </body>
        </html>
        """
        file.write(next_str)

        file.close()

    def html_report_3(self):
        """
        does: creates an .html file that makes the first report (second app).
        """
        file = open(u"/kb/module/work/tmp/forHTML/nice_html3.html", u"w")

        html_string = u"""
        <!DOCTYPE html>
        <html>
        <head>
        <style>
        table, th, td {
            border: 1px solid black;
            border-collapse: collapse;
        }

        * {
            box-sizing: border-box;
        }

        .column {
            float: left;
            width: 50%;
            padding: 10px;
        }

        /* Clearfix (clear floats) */
        .row::after {
            content: "";
            clear: both;
            display: table;
        }
        </style>
        </head>
        <body>

        <h1 style="text-align:center;">Prediction of Classifications</h1>
        """
        file.write(html_string)

        another_file = open(u'/kb/module/work/tmp/results.html', u"r")
        all_str = another_file.read()
        another_file.close()

        file.write(all_str)

        next_str= u"""
        </body>
        </html>
        """
        file.write(next_str)

        file.close()

    def printTree(self,tree, pass_name):
        """
        args:
        ---tree is a DecisionTree object that has already been tuned
        ---pass_name is a string for what you want the tree named as (but this is not where the creation happens just pass)
        does:
        ---using graphviz feature it is able to geneate the dot file that has an "ugly" version of the tree inside
        ---call the parse_lookNice
        return:
        ---N/A just makes an "ugly" dot file.
        """

        """
        export_graphviz(tree, out_file="mytree.dot", feature_names=self.attribute_list,
                        class_names=self.class_list)
        with open("mytree.dot") as f:
            dot_graph = f.read()
        os.system('dot -Tpng mytree.dot >  ' + name + '.png ')
        """

        #dotfile = io.open(u"/kb/module/work/tmp/dotFolder/mydotTree.dot", u'w')
        not_dotfile = StringIO.StringIO()
        export_graphviz(tree, out_file=not_dotfile, feature_names=self.attribute_list,
                        class_names=self.class_list)
        #dotfile.close()

        #print(type(dotfile))
        #print(dotfile.getvalue())
        contents = not_dotfile.getvalue()
        not_dotfile.close()

        dotfile = open(u"/kb/module/work/tmp/dotFolder/mydotTree.dot", u'w')
        dotfile.write(contents)
        dotfile.close()

        self.parse_lookNice(pass_name)
        #os.system('dot -Tpng ./mytree.dot >  ' + name + '.png ')

    def create_report(self):
        """
        at the moment not being used
        """
        uuid_string = str(uuid.uuid4())

        report_params = {
            'direct_html_link_index': 0,
            'file_links': output_zip_files,
            'html_links': [u"/kb/module/work/tmp/forHTML/nice_html1.html", u"/kb/module/work/tmp/forHTML/nice_html2.html"],
            'workspace_name': ws,
            'report_object_name': 'kb_classifier_report_' + uuid_string
        }

        kbase_report_client = KBaseReport(self.callback_url, token=token)
        output = kbase_report_client.create_extended_report(report_params)
        return output

    def get_mainAttributes(self,my_input, my_current_ws, for_predict = False):
        """
        args:
        ---my_input is either a list of the names of the genomes in format "name1,name2" or "all" meaning everything in workspace will get used
        does:
        ---creates a dataframe for the all the genomes given
            ---Rows are "index" which is the name of the genome(same as my_input)
            ---Colmuns are "master Role" which is a list of the all functional roles
        return:
        ---returns the dataframe which contains all_attributes (this is the X matrix for ML)
        """

        #current_ws = os.environ['KB_WORKSPACE_ID']
        print my_input

        current_ws = my_current_ws

        print current_ws
        #ws = biokbase.narrative.clients.get("workspace")
        #ws_client = Workspace()

        #ws_client = workspaceService(config["workspace-url"])

        listOfNames = [] #make this self.listOfNames
        
        if not for_predict:
            master_Role = [] #make this master_Role


        name_and_roles = {}

        #my_input = 'all' # change this to something that is passed in the input

        if my_input == 'all':
            wsgenomes = self.ws_client.list_objects({"workspaces":[current_ws],"type":"KBaseGenomes.Genome"});
            for genome in wsgenomes:
                listOfNames.append(str(genome[1]))

        else:
            listOfNames = my_input.split(',')

        for current_gName in listOfNames:
            listOfFunctionalRoles = []
            try:
                functionList = self.ws_client.get_objects([{'workspace':current_ws, 'name':current_gName}])[0]['data']['cdss']
                for function in range(len (functionList)):
                    if str(functionList[function]['functions'][0]).lower() != 'hypothetical protein':
                        listOfFunctionalRoles.append(str(functionList[function]['functions'][0]))

            except:
                functionList = self.ws_client.get_objects([{'workspace':current_ws, 'name':current_gName}])[0]['data']['features']
                for function in range(len (functionList)):
                    if str(functionList[function]['function']).lower() != 'hypothetical protein':
                        listOfFunctionalRoles.append(str(functionList[function]['function']))

            name_and_roles[current_gName] = listOfFunctionalRoles

            print "I have arrived inside the desired for loop!!"

        if not for_predict:
            master_pre_Role = list(itertools.chain(*name_and_roles.values()))
            master_Role = list(set(master_pre_Role))


        data_dict = {}

        for current_gName in listOfNames:
            arrayofONEZERO = []

            current_Roles = name_and_roles[current_gName]

            for individual_role in master_Role:
                if individual_role in current_Roles:
                    arrayofONEZERO.append(1)
                else:
                    arrayofONEZERO.append(0)

            data_dict[current_gName] = arrayofONEZERO

        my_all_attributes = pd.DataFrame.from_dict(data_dict, orient='index', columns = master_Role)

        return my_all_attributes, master_Role

    def get_mainClassification(self, file_path):
        """
        args:
        ---file_path is a path that holds the path of where the excel file is located (given as input by the user)
        does:
        ---with the excel file which has 2 columns: Genome_ID (same as my_input) and Classification
            ---it creates another dataframe with only classifications and rows as "index" which are genome names (my_input)
        return:
        ---the dataframe with all_classifications (essentially the Y variable for ML)
        """

        #figure out how to read in xls file
        my_all_classifications = pd.read_excel(file_path) #replace with location of file
        my_all_classifications.set_index('Genome_ID', inplace=True)

        print my_all_classifications

        return my_all_classifications

    def _valid_params(self, params):

        pass

    def _make_dir(self):
        dir_path = os.path.join(self.scratch, str(uuid.uuid4()))
        # if os # exists
        os.mkdir(dir_path)

        return dir_path

    def _download_shock(self, shock_id):
        dir_path = self._make_dir()

        file_path = self.dfu.shock_to_file({'shock_id': shock_id,
                                            'file_path': dir_path})['file_path']

        return file_path

    #END_CLASS_HEADER

    # config contains contents of config file in a hash or None if it couldn't
    # be found
    def __init__(self, config):
        #BEGIN_CONSTRUCTOR

        """
        # Any configuration parameters that are important should be parsed and
        # saved in the constructor.
        self.callback_url = os.environ['SDK_CALLBACK_URL']
        self.shared_folder = config['scratch']

        self.full_attribute_array = np.load("/kb/module/data/full_attribute_array.npy")
        self.full_classification_array = np.load("/kb/module/data/full_classification_array.npy")

        pickle_in = open("/kb/module/data/attribute_list.pickle", "rb")
        self.attribute_list = pickle.load(pickle_in)

        self.class_list = ['anaerobic', 'aerobic', 'facultative']

        self.train_index = []
        self.test_index = []

        splits = 10

        self.classifier_name = ""
        """
        #which_target = u"Metabolism"

        self.workspaceURL = config.get('workspace-url')
        self.scratch = os.path.abspath(config.get('scratch'))
        self.callback_url = os.environ['SDK_CALLBACK_URL']
        self.dfu = DataFileUtil(self.callback_url)
        self.ws_client = workspaceService(self.workspaceURL)

        """
        which_target = u"Gram_Stain"

        print "I am right here"

        if which_target == u"Metabolism":
            self.full_attribute_array = np.load(u"/kb/module/data/full_attribute_array.npy")
            self.full_classification_array = np.load(u"/kb/module/data/full_classification_array.npy")
            self.class_list = [u'anaerobic', u'aerobic', u'facultative']

        if which_target == u"Gram_Stain":
            self.full_attribute_array = np.load(u"/kb/module/data/copyof_gram_full_attribute_array.npy")
            self.full_classification_array = np.load(u"/kb/module/data/copyof_gram_full_classification_array.npy")
            self.class_list = [u'N',u'P']


        pickle_in = open(u"/kb/module/data/attribute_list.pickle", u"rb")
        self.attribute_list = pickle.load(pickle_in)
        """

        #self.global_target = which_target

        self.global_target = ''
        
        self.list_name = []

        self.list_statistics = []

        #global output 
        #output = {'jack': 4098, 'sape': 4139} #random dict

        #END_CONSTRUCTOR
        pass


    def build_classifier(self, ctx, params):
        """
        build_classifier: build_classifier
        requried params:
        :param params: instance of type "BuildClassifierInput" -> structure:
           parameter "phenotypeclass" of String, parameter "attribute" of
           String, parameter "workspace" of String, parameter
           "classifier_training_set" of mapping from String to type
           "ClassifierTrainingSet" (typedef string genome_id; typedef string
           phenotype;) -> structure: parameter "phenotype" of String,
           parameter "genome_name" of String, parameter "classifier_out" of
           String, parameter "target" of String, parameter "classifier" of
           String, parameter "shock_id" of String, parameter "list_name" of
           String, parameter "save_ts" of Long
        :returns: instance of type "ClassifierOut" -> structure: parameter
           "report_name" of String, parameter "report_ref" of String
        """
        # ctx is the context object
        # return variables are: output
        #BEGIN build_classifier

        print params

        file_path = self._download_shock(params.get('shock_id'))
        #file_path = '/kb/module/data/newTrialRun.xlsx'

        #current_ws janakakbase:narrative_1533153056355

        all_attributes, master_Role = self.get_mainAttributes(params.get('list_name'), params.get('workspace'))
        all_classifications = self.get_mainClassification(file_path)

        full_dataFrame = pd.concat([all_attributes, all_classifications], axis = 1, sort=True)

        print full_dataFrame

        #should include self??
        class_list = list(set(full_dataFrame['Classification']))

        self.class_list = class_list

        #create a mapping
        my_mapping = {}
        for current_class,num in zip(class_list, range(0, len(class_list))):
            my_mapping[current_class] = num

        for index in full_dataFrame.index:
            full_dataFrame.at[index, 'Classification'] = my_mapping[full_dataFrame.at[index, 'Classification']]

        all_classifications = full_dataFrame['Classification']

        self.full_attribute_array = all_attributes
        self.full_classification_array = all_classifications

        print self.full_attribute_array
        print self.full_classification_array


        self.full_attribute_array = self.full_attribute_array.values.astype(int)
        self.full_classification_array = self.full_classification_array.values.astype(int)

        print self.full_attribute_array
        print self.full_classification_array

        os.makedirs("/kb/module/work/tmp/pics/")
        os.makedirs("/kb/module/work/tmp/dotFolder/")
        os.makedirs("/kb/module/work/tmp/forHTML/")
        os.makedirs("/kb/module/work/tmp/forDATA/")

        print 'fdsafds'
        print params

        token = ctx['token']
        # wsClient = workspaceService(self.workspaceURL, token=token)

        self._valid_params(params)

        classifier = params.get('classifier')
        target = params.get('phenotypeclass')

        self.global_target = target

        # Add the block of code that reads in .txt file contain the annotations.
        # (Here my question is how to change this section so that it reads in the genomes files on the KBASE Narrative)

        train_index = []
        test_index = []

        splits = 2 #10


        skf = StratifiedKFold(n_splits=splits, random_state=0, shuffle=True)
        for train_idx, test_idx in skf.split(self.full_attribute_array, self.full_classification_array):
            train_index.append(train_idx)
            test_index.append(test_idx)

        if classifier == u"run_all":
            self.classifierTest(KNeighborsClassifier(),target+u"_KNeighborsClassifier", my_mapping, master_Role, splits, train_index, test_index,True)
            self.classifierTest(GaussianNB(),target+u"_GaussianNB", my_mapping, master_Role, splits, train_index, test_index,True)
            self.classifierTest(LogisticRegression(random_state=0),target+u"_LogisticRegression", my_mapping, master_Role, splits, train_index, test_index,True)
            self.classifierTest(DecisionTreeClassifier(random_state=0),target+u"_DecisionTreeClassifier", my_mapping, master_Role, splits, train_index, test_index,True)
            self.classifierTest(svm.LinearSVC(random_state=0),target+u"_SVM", my_mapping, master_Role, splits, train_index, test_index,True)
            self.classifierTest(MLPClassifier(random_state=0),target+u"_NeuralNetwork", my_mapping, master_Role, splits, train_index, test_index, True)
        else:
            if target == u"Metabolism":
                self.classifierTest(self.whichClassifier(classifier), unicode(u"Metabolism_") + classifier, my_mapping, master_Role, splits, train_index, test_index, True)
            elif target == u"Gram_Stain":
                self.classifierTest(self.whichClassifier(classifier), unicode(u"Gram_Stain_") + classifier, my_mapping, master_Role, splits, train_index, test_index, True)
            else:
                print u"ERROR check spelling?"

            #self.classifierTest(self.whichClassifier(classifier), unicode(target + u"_") + classifier, True)



        best_classifier_str = self.to_HTML_Statistics()
        self.html_report_1(classifier, best_classifier_str)

        #self.tune_Decision_Tree(splits, train_index, test_index)


        #self.tree_code("doesn't matter") #<-- don't use rn

        #self.html_report_2()


        uuid_string = str(uuid.uuid4())

        output_directory = '/kb/module/work/tmp/forHTML'

        report_shock_id = self.dfu.file_to_shock({'file_path': output_directory,
                                                  'pack': 'zip'})['shock_id']


        htmloutput1 = {
        'description' : 'htmloutuput1description',
        'name' : 'html1.html',
        'label' : 'htmloutput1label',
        'shock_id': report_shock_id
        }

        
        htmloutput2 = {
        'description' : 'htmloutuput2description',
        'name' : 'html2.html',
        'label' : 'htmloutput2label',
        'shock_id': report_shock_id
        }

        """
        fileoutput1 = {
        'description' : 'htmloutuput2description',
        'name' : 'htmloutput2name',
        'label' : 'htmloutput2label',
        'URL' : "/kb/module/work/tmp/forDATA/" + self.best_classifier_str + u".pickle"
        }
        """

        report_params = {'message': '',
                         'workspace_name': params.get('workspace'),#params.get('input_ws'),
                         #'objects_created': objects_created,
                         'html_links': [htmloutput1, htmloutput2],
                         'direct_html_link_index': 0,
                         'html_window_height': 333,
                         'report_object_name': 'kb_classifier_report_' + str(uuid.uuid4())}

        kbase_report_client = KBaseReport(self.callback_url, token=token)
        report_output = kbase_report_client.create_extended_report(report_params)

        output = {'report_name': report_output['name'], 'report_ref': report_output['ref']}

        print('I hope I am working now - this means that I am past the report generation')

        print(output.get('report_name')) # kb_classifier_report_5920d1da-2a99-463b-94a5-6cb8721fca45
        print(output.get('report_ref')) #19352/1/1

        #return output

        #return report_output

        #END build_classifier

        # At some point might do deeper type checking...
        if not isinstance(report_output, dict):
            raise ValueError('Method build_classifier return value ' +
                             'output is not type dict as required.')
        # return the results
        return [report_output]

    def predict_phenotype(self, ctx, params):
        """
        :param params: instance of type "ClassifierPredictionInput" ->
           structure: parameter "workspace" of String, parameter
           "classifier_ref" of String, parameter "phenotype" of String
        :returns: instance of type "ClassifierPredictionOutput" -> structure:
           parameter "prediction_accuracy" of Double, parameter "predictions"
           of mapping from String to String
        """
        # ctx is the context object
        # return variables are: output
        #BEGIN predict_phenotype

        classifier_name_rn = best_classifier_str
        #this can be passed in as a key value that the user can select

        with open(u"/kb/module/work/tmp/" + unicode(classifier_name_rn) + u".txt", u"r") as f:
            still_str = f.read()

        after_classifier = pickle.loads(codecs.decode(still_str.encode(), "base64"))
        #should be able to run regular commands like... after_classifier.predict(insert_X)

        """
        all_attributes = self.get_mainAttributes(params.get('list_name'), for_predict = True)
        #again here we need to edit how to feed in the inputs
        """

        after_classifier_result = after_classifier.predict(all_attributes) #replace with all_attributes

        after_classifier_result_forDF = []

        for current_result in after_classifier_result:
            after_classifier_result_forDF.extend(my_mapping.keys()[my_mapping.values().index(current_result)])

        after_classifier_df = pd.DataFrame(after_classifier_result_forDF, index=all_attributes.index)


        allProbs = after_classifier.predict_proba(self.full_attribute_array)
        maxEZ = np.amax(allProbs, axis=1)
        maxEZ_df = pd.DataFrame(maxEZ, index=all_attributes.index)

        predict_table_pd = pd.concat([after_classifier_df, maxEZ_df], axis=1)
        predict_table_pd.to_html(u'/kb/module/work/tmp/forHTML/results.html')

        #you can also save down table as text file or csv
        """
        #txt
        np.savetxt(r'/kb/module/work/tmp/np.txt', predict_table_pd.values, fmt='%d')

        #csv
        predict_table_pd.to_csv(r'/kb/module/work/tmp/pandas.txt', header=None, index=None, sep=' ', mode='a')
        """

        self.html_report_3()

        #END predict_phenotype

        # At some point might do deeper type checking...
        if not isinstance(output, dict):
            raise ValueError('Method predict_phenotype return value ' +
                             'output is not type dict as required.')
        # return the results
        return [output]
    def status(self, ctx):
        #BEGIN_STATUS
        returnVal = {'state': "OK",
                     'message': "",
                     'version': self.VERSION,
                     'git_url': self.GIT_URL,
                     'git_commit_hash': self.GIT_COMMIT_HASH}
        #END_STATUS
        return [returnVal]
