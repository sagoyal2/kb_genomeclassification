# -*- coding: utf-8 -*-
#BEGIN_HEADER
# The header block is where all import statments should live
import os
from Bio import SeqIO
from pprint import pprint, pformat
from AssemblyUtil.AssemblyUtilClient import AssemblyUtil
from KBaseReport.KBaseReportClient import KBaseReport
#END_HEADER


#placing here to test branch versions




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
    #END_CLASS_HEADER

    # config contains contents of config file in a hash or None if it couldn't
    # be found
    def __init__(self, config):
        #BEGIN_CONSTRUCTOR
        
        # Any configuration parameters that are important should be parsed and
        # saved in the constructor.
        self.callback_url = os.environ['SDK_CALLBACK_URL']
        self.shared_folder = config['scratch']

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
        #END build_classifier

        # At some point might do deeper type checking...
        if not isinstance(output, dict):
            raise ValueError('Method build_classifier return value ' +
                             'output is not type dict as required.')
        # return the results
        return [output]

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
