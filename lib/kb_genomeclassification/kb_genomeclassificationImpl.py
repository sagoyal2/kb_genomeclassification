#BEGIN_HEADER
# The header block is where all import statments should live
import os
from Bio import SeqIO
from pprint import pprint, pformat
from AssemblyUtil.AssemblyUtilClient import AssemblyUtil
from KBaseReport.KBaseReportClient import KBaseReport

    #here are some more imports for sklearn
from sklearn.model_selection import train_test_split
#from sklearn.grid_search import train_test_split
from sklearn.feature_selection import SelectKBest
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import feature_selection
from sklearn.metrics import recall_score
from sklearn.model_selection import StratifiedKFold
#from sklearn.grid_search import StratifiedKFold

#fix later
#import seaborn; seaborn.set()
#import matplotlib.pyplot as plt

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
    GIT_URL = ""
    GIT_COMMIT_HASH = ""

    #BEGIN_CLASS_HEADER
    # Class variables and functions can be defined in this block

    def classiferTest(self, classifier, classifier_name, print_cfm):
        # so in this method we need to be returning a classifier instead of all the confusion matrices?
        # how do you return a classifier?

        if print_cfm:
            print(classifier_name)
        train_score = numpy.zeros(self.splits)
        validate_score = numpy.zeros(self.splits)
        cnf_matrix = numpy.zeros(shape=(3, 3))
        cnf_matrix_f = numpy.zeros(shape=(3, 3))
        for c in range(self.splits):
            X_train = self.full_attribute_array[self.train_index[c]]
            y_train = self.full_classification_array[self.train_index[c]]
            X_test = self.full_attribute_array[self.test_index[c]]
            y_test = self.full_classification_array[self.test_index[c]]
            classifier.fit(X_train, y_train)
            train_score[c] = classifier.score(X_train, y_train)
            validate_score[c] = classifier.score(X_test, y_test)
            y_pred = classifier.predict(X_test)
            cnf = confusion_matrix(y_test, y_pred)
            cnf_f = cnf.astype('float') / cnf.sum(axis=1)[:, numpy.newaxis]
            for i in range(len(cnf)):
                for j in range(len(cnf)):
                    cnf_matrix[i][j] += cnf[i][j]
                    cnf_matrix_f[i][j] += cnf_f[i][j]

        print("%6.3f\t%6.3f\t%6.3f\t%6.3f" % (
        numpy.average(train_score), numpy.std(train_score), numpy.average(validate_score), numpy.std(validate_score)))

        if print_cfm:
            cnf_av = cnf_matrix / self.splits
            print()
            print(cnf_av[0][0], cnf_av[0][1], cnf_av[0][2], )
            print(cnf_av[1][0], cnf_av[1][1], cnf_av[1][2], )
            print(cnf_av[2][0], cnf_av[2][1], cnf_av[2][2], )
            print()
            print(self.class_list[0])
            TP = cnf_av[0][0]
            TN = cnf_av[1][2] + cnf_av[1][2] + cnf_av[2][1] + cnf_av[2][2]
            FP = cnf_av[0][1] + cnf_av[0][2]
            FN = cnf_av[1][0] + cnf_av[2][0]
            self.cf_stats(TN, TP, FP, FN)

            print(self.class_list[1])
            TP = cnf_av[1][1]
            TN = cnf_av[0][0] + cnf_av[0][2] + cnf_av[2][0] + cnf_av[2][2]
            FP = cnf_av[1][0] + cnf_av[1][2]
            FN = cnf_av[0][1] + cnf_av[2][1]
            self.cf_stats(TN, TP, FP, FN)

            print(self.class_list[2])
            TP = cnf_av[2][2]
            TN = cnf_av[0][0] + cnf_av[0][1] + cnf_av[1][0] + cnf_av[1][1]
            FP = cnf_av[2][0] + cnf_av[2][1]
            FN = cnf_av[0][1] + cnf_av[0][2]
            self.cf_stats(TN, TP, FP, FN)

            print(classifier)
            print()
            """
            print("Confusion matrix")
            for i in range(len(cnf_matrix)):
                print(self.class_list[i], end="  \t")
                for j in range(len(cnf_matrix[i])):
                    print(cnf_matrix[i][j] / self.splits, end="\t")
                print()
            print()
            for i in range(len(cnf_matrix_f)):
                print(self.class_list[i], end="  \t")
                for j in range(len(cnf_matrix_f[i])):
                    print("%6.1f" % ((cnf_matrix_f[i][j] / self.splits) * 100.0), end="\t")
                print()
            print()
            print("01", cnf_matrix[0][1])
            """

            print("Confusion matrix")
            for i in range(len(cnf_matrix)):
                print(str(self.class_list[i]) + "  \t"),
                for j in range(len(cnf_matrix[i])):
                    print(str(cnf_matrix[i][j] / self.splits) + "\t"),
                print()
            print()
            for i in range(len(cnf_matrix_f)):
                print(str(self.class_list[i]) + "  \t"),
                for j in range(len(cnf_matrix_f[i])):
                    print(str("%6.1f" % ((cnf_matrix_f[i][j] / self.splits) * 100.0)) + "\t"),
                print()
            print()
            print("01", cnf_matrix[0][1])


            # plot_confusion_matrix(cnf_matrix/10,class_list,'Confusion Matrix')
            # plot_confusion_matrix(cnf_matrix_f/splits*100.0,class_list,'Confusion Matrix %',classifier_name)
        return ( numpy.average(train_score), numpy.std(train_score), numpy.average(validate_score), numpy.std(validate_score))

    def cf_stats(self,TN,TP,FP,FN):
        AN = TN+FP
        AP = TN+FN
        PN = TN+FN
        PP = TP+FP
        Total = TN+TP+FP+FN
        Recall = (TP/(TP+FN))
        Precision = (TP/(TP+FP))
        print("Accuracy:\t\t%6.3f"%((TP+TN)/Total))
        print("Precision:\t\t%6.3f"%(Precision))
        print("Recall:\t\t%6.3f"%(Recall))
        print("F1 score::\t\t%6.3f"%(2*((Precision*Recall)/(Precision+Recall))))
        print()

    """
    def plot_confusion_matrix(cm, classes, title,classifier_name):
        plt.rcParams.update({'font.size': 18})
        fig,ax= plt.subplots(figsize=(5,4))
        sns.set(font_scale=1.5)
        sns_plot = sns.heatmap(cm, annot=True, ax = ax, cmap="Blues"); #annot=True to annotate cells
        # labels, title and ticks
        ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
        ax.set_title(title); 
        ax.xaxis.set_ticklabels(classes); ax.yaxis.set_ticklabels(classes);
        # fig.savefig(classifier_name+".png") this may not be necessary as not necessary to save png file
    """

    """
    def tree_code(tree, feature_names, target_names,
                 spacer_base="    "):

        
        (below was previously placed inside the build_classifier)
        #building the visual tree for the DecisionTreeClassifier
        #tree = DecisionTreeClassifier(random_state=0,max_depth=3, criterion='entropy')
        #tree.fit(full_attribute_array,full_classification_array)
        #tree_code(tree, attribute_list, class_list) # <-- prints the tree to see the "decisions" that the classifier is making
        


        Produce psuedo-code for decision tree.
        Notes
        -----
        based on http://stackoverflow.com/a/30104792.
            
        left      = tree.tree_.children_left
        right     = tree.tree_.children_right
        threshold = tree.tree_.threshold
        features  = [feature_names[i] for i in tree.tree_.feature]
        value = tree.tree_.value

        def recurse(left, right, threshold, features, node, depth):
            spacer = spacer_base * depth
            if (threshold[node] != -2):
                print(spacer + "if ( " + features[node] + " <= " + \
                      str(threshold[node]) + " ) {")
                if left[node] != -1:
                        recurse(left, right, threshold, features,
                                left[node], depth+1)
                print(spacer + "}\n" + spacer +"else {")
                if right[node] != -1:
                        recurse(left, right, threshold, features,
                                right[node], depth+1)
                print(spacer + "}")
            else:
                target = value[node]
                for i, v in zip(numpy.nonzero(target)[1],
                                target[numpy.nonzero(target)]):
                    target_name = target_names[i]
                    target_count = int(v)
                    print(spacer + "return " + str(target_name) + \
                          " ( " + str(target_count) + " examples )")

        recurse(left, right, threshold, features, 0, 0)
    """

    def tree_code(self, spacer_base="    "):
        """Produce psuedo-code for decision tree.
        Notes
        -----
        based on http://stackoverflow.com/a/30104792.
        """
        tree = DecisionTreeClassifier(random_state=0, max_depth=3, criterion='entropy')

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
                print(spacer + "if ( " + features[node] + " <= " + \
                      str(threshold[node]) + " ) {")
                if left[node] != -1:
                    recurse(left, right, threshold, features,
                                 left[node], depth + 1)
                print(spacer + "}\n" + spacer + "else {")
                if right[node] != -1:
                    recurse(left, right, threshold, features,
                                 right[node], depth + 1)
                print(spacer + "}")
            else:
                target = value[node]
                for i, v in zip(numpy.nonzero(target)[1],
                                target[numpy.nonzero(target)]):
                    target_name = target_names[i]
                    target_count = int(v)
                    print(spacer + "return " + str(target_name) + \
                          " ( " + str(target_count) + " examples )")

        recurse(left, right, threshold, features, 0, 0)

        return("reached the end of the tree_code")

    #END_CLASS_HEADER

    # config contains contents of config file in a hash or None if it couldn't
    # be found
    def __init__(self, config):
        #BEGIN_CONSTRUCTOR
        
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

        self.splits = 10

        global output 
        output = {'jack': 4098, 'sape': 4139} #random dict

        #END_CONSTRUCTOR
        pass

    def build_classifier(self, ctx, params):
        """
        :param params: instance of type "BuildClassifierInput" -> structure:
           parameter "phenotypeclass" of String, parameter "attribute" of
           String, parameter "workspace" of String, parameter
           "classifier_training_set" of mapping from String to type
           "ClassifierTrainingSet" (typedef string genome_id; typedef string
           phenotype;) -> structure: parameter "phenotype" of String,
           parameter "genome_name" of String
        :returns: instance of type "ClassifierOut" -> structure: parameter
           "classifier_ref" of String, parameter "phenotype" of String
        """
        # ctx is the context object
        # return variables are: output
        #BEGIN build_classifier

        print(params)

        # Add the block of code that reads in .txt file contain the annotations.
        # (Here my question is how to change this section so that it reads in the genomes files on the KBASE Narrative)
        """
        organisams={}
        organism_id=[]
        attributes_full={}
        atributes_by_class={}
        class_set=set()
        class_counter = {'anaerobic':0, 'aerobic':0, 'facultative':0}
        with open("/kb/module/data/cleanData.txt") as infile:
            for line in infile:
                # data is tab delimited
                data = line.rstrip().split("\t")
                # first column contains organism name
                organism_id.append(data[0])
                # second column contains metabolic class
                organisams[data[0]]={"class":data[1],"roles":[]}
                class_set.add(data[1].replace("'", ""))
                classtype=data[1].replace("'", "")
                class_counter[classtype]+=1
                for i in range(2,len(data)):
                    attribute = data[i].replace("'", "")
                    if(attribute not in organisams[data[0].replace("'", "")]["roles"]):
                        organisams[data[0]]["roles"].append(attribute)
                        if not attribute in attributes_full:
                            attributes_full[attribute]=0
                            atributes_by_class[attribute]={'anaerobic':0, 'aerobic':0, 'facultative':0}
                        attributes_full[attribute]+=1
                        atributes_by_class[attribute][classtype]+=1

        print(len(organisams))
        print(len(attributes_full))
        print(class_set)
        print(class_counter)

        # Add in block of code that creates one dimensional array is constructed for the organisms and all of the functional roles.
        import numpy
        attribute_list = list(attributes_full.keys())
        organisam_list = list(organisams.keys())
        class_list = list(class_set)
        full_classification_array = numpy.array(["" for x in range(len(organisam_list))])

        print(class_list)
        print(len(attributes_full))
        print(len(organisam_list))
        print(len(full_classification_array))

        full_attribute_array = numpy.zeros(shape=(len(organisam_list),len(attribute_list)))

        # Add in block of code used to create the functional role data matrix; full_attribute_array

        for y_indx in range(len(organisam_list)):
            organisam = organisam_list[y_indx]
            if organisam in organisams:
                full_classification_array[y_indx] = class_list.index(organisams[organisam]["class"])
                for attribute in organisams[organisam]["roles"]:
                    if attribute in attributes_full:
                        x_indx  = attribute_list.index(attribute)
                        full_attribute_array[y_indx,x_indx]=1
        """


        skf = StratifiedKFold(n_splits=self.splits, random_state=0, shuffle=True)
        for train_idx, test_idx in skf.split(self.full_attribute_array, self.full_classification_array):
            self.train_index.append(train_idx)
            self.test_index.append(test_idx)
        
        self.classiferTest(KNeighborsClassifier(),"Metabolism-KNeighborsClassifier",True)

        print("this is the last line of the build_classifier")

        #END build_classifier

        # At some point might do deeper type checking...
        if not isinstance(output, dict):
            raise ValueError('Method build_classifier return value ' +
                             'output is not type dict as required.')
        # return the results
        return [output]

    """
    def predict_phenotype(self, ctx, params):
        :param params: instance of type "ClassifierPredictionInput" ->
           structure: parameter "workspace" of String, parameter
           "classifier_ref" of String, parameter "phenotype" of String
        :returns: instance of type "ClassifierPredictionOutput" -> structure:
           parameter "prediction_accuracy" of Double, parameter "predictions"
           of mapping from String to String
        # ctx is the context object
        # return variables are: output
        #BEGIN predict_phenotype
        #END predict_phenotype

        # At some point might do deeper type checking...
        if not isinstance(output, dict):
            raise ValueError('Method predict_phenotype return value ' +
                             'output is not type dict as required.')
        # return the results
        return [output]
    """

    def status(self, ctx):
        #BEGIN_STATUS
        returnVal = {'state': "OK",
                     'message': "",
                     'version': self.VERSION,
                     'git_url': self.GIT_URL,
                     'git_commit_hash': self.GIT_COMMIT_HASH}
        #END_STATUS
        return [returnVal]
