# -*- coding: utf-8 -*-
#BEGIN_HEADER
# The header block is where all import statments should live

import os
from kb_genomeclassification.Util.kb_genomeclfUtils import kb_genomeclfUtils

#END_HEADER


class kb_genomeclassification:
    '''
    Module Name:
    kb_genomeclassification

    Module Description:
    A KBase module: kb_genomeclassification
This module build a classifier and predict phenotypes based on the classifier Another line
    '''

    ######## WARNING FOR GEVENT USERS ####### noqa
    # Since asynchronous IO can lead to methods - even the same method -
    # interrupting each other, you must be *very* careful when using global
    # state. A method could easily clobber the state set by another while
    # the latter method is running.
    ######################################### noqa
    VERSION = "0.0.1"
    GIT_URL = "https://github.com/janakagithub/kb_genomeclassification.git"
    GIT_COMMIT_HASH = "6a61599c51d991c92f26f26252ca3d109b8ceb42"

    #BEGIN_CLASS_HEADER

    #END_CLASS_HEADER

    # config contains contents of config file in a hash or None if it couldn't
    # be found
    def __init__(self, config):
        #BEGIN_CONSTRUCTOR

        self.config = config

        self.workspaceURL = config.get('workspace-url')
        self.scratch = os.path.abspath(config.get('scratch'))
        self.callback_url = os.environ['SDK_CALLBACK_URL']

        self.config['workspaceURL'] = self.workspaceURL
        self.config['scratch'] = self.scratch
        self.config['callback_url'] = self.callback_url

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

        self.config['ctx'] = ctx
        clf_Runner = kb_genomeclfUtils(self.config)

        location_of_report = clf_Runner.fullClassify(params, params.get('workspace'))

        report_output = clf_Runner.makeHtmlReport(location_of_report, params.get('workspace'), 'clf_Runner')
    	output = {'report_name': report_output['name'], 'report_ref': report_output['ref']}

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
           "classifier_name" of String, parameter "phenotypeclass" of String,
           parameter "shock_id" of String, parameter "list_name" of String
        :returns: instance of type "ClassifierPredictionOutput" -> structure:
           parameter "prediction_accuracy" of Double, parameter "predictions"
           of mapping from String to String
        """
        # ctx is the context object
        # return variables are: output
        #BEGIN predict_phenotype

        print params

        self.config['ctx'] = ctx
        pred_Runner = kb_genomeclfUtils(self.config)

        location_of_report = pred_Runner.fullPredict(params, params.get('workspace'))

        report_output = pred_Runner.makeHtmlReport(location_of_report, params.get('workspace'), 'pred_Runner')
    	output = {'report_name': report_output['name'], 'report_ref': report_output['ref']}

        #END predict_phenotype

        # At some point might do deeper type checking...
        if not isinstance(output, dict):
            raise ValueError('Method predict_phenotype return value ' +
                             'output is not type dict as required.')
        # return the results
        return [output]

    def upload_trainingset(self, ctx, params):
        """
        :param params: instance of type "UploadTrainingSetInput" ->
           structure: parameter "phenotypeclass" of String, parameter
           "workspace" of String, parameter "classifier_training_set" of
           mapping from String to type "ClassifierTrainingSet" (typedef
           string genome_id; typedef string phenotype;) -> structure:
           parameter "phenotype" of String, parameter "genome_name" of
           String, parameter "training_set_out" of String, parameter "target"
           of String, parameter "shock_id" of String, parameter "list_name"
           of String
        :returns: instance of type "UploadTrainingSetOut" -> structure:
           parameter "report_name" of String, parameter "report_ref" of String
        """
        # ctx is the context object
        # return variables are: output
        #BEGIN upload_trainingset
        #END upload_trainingset

        # At some point might do deeper type checking...
        if not isinstance(output, dict):
            raise ValueError('Method upload_trainingset return value ' +
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
