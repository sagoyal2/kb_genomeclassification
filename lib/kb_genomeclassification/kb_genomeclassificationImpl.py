# -*- coding: utf-8 -*-
#BEGIN_HEADER
# The header block is where all import statments should live

import os

from biokbase.workspace.client import Workspace as workspaceService
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
    GIT_COMMIT_HASH = "346e01dd28a6eb6b08f577650fc634904b83586f"

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
        self.ws_client = workspaceService(self.workspaceURL)

        self.config['workspaceURL'] = self.workspaceURL
        self.config['scratch'] = self.scratch
        self.config['callback_url'] = self.callback_url

        #END_CONSTRUCTOR

    def upload_trainingset(self, ctx, params):
        """
        :param
        :returns
        """
        # ctx is the context object
        # return variables are: output
        #BEGIN upload_trainingset

        self.config['ctx'] = ctx
        upload_Runner = kb_genomeclfUtils(self.config)
        print("in here")
        upload_Runner.fullUpload(params, params.get('workspace'))
        html_output_name, classifier_training_set = upload_Runner.fullUpload(params, params['workspace'])

        report_output = upload_Runner.generateHTMLReport(params['workspace'], "forUpload", html_output_name, params['description'])
        output = {'report_name': report_output['name'], 'report_ref': report_output['ref'], 'classifier_training_set': classifier_training_set}

        #END upload_trainingset

        # # At some point might do deeper type checking...
        # if not isinstance(output, dict):
        #     raise ValueError('Method upload_trainingset return value ' +
        #                      'output is not type dict as required.')
        # return the results
        return [output]

    def build_classifier(self, ctx, params):
        """
        build_classifier: build_classifier
        requried params:
        :param 
        :returns
        """
        # ctx is the context object
        # return variables are: output
        #BEGIN build_classifier

        # self.config['ctx'] = ctx
        # clf_Runner = kb_genomeclfUtils(self.config)

        # location_of_report, classifier_info_list, attribute_weights_list = clf_Runner.fullClassify(params, params.get('workspace'))
        # report_output = clf_Runner.makeHtmlReport(location_of_report, params.get('workspace'), 'clf_Runner', params.get('description'))
        # output = {'report_name': report_output['name'], 'report_ref': report_output['ref'], 'classifier_info': classifier_info_list, 'attribute_weights': attribute_weights_list }

        #END build_classifier

        # At some point might do deeper type checking...
        if not isinstance(output, dict):
            raise ValueError('Method build_classifier return value ' +
                             'output is not type dict as required.')
        # return the results
        # return [output]
        return

    def predict_phenotype(self, ctx, params):
        """
        :param
        :returns
        """
        # ctx is the context object
        # return variables are: output
        #BEGIN predict_phenotype
        # self.config['ctx'] = ctx
        # pred_Runner = kb_genomeclfUtils(self.config)
        # location_of_report, predictions_mapping = pred_Runner.fullPredict(params, params.get('workspace'))
        # report_output = pred_Runner.makeHtmlReport(location_of_report, params.get('workspace'), 'pred_Runner', params.get('description'), for_predict = True)
        # output = {'report_name': report_output['name'], 'report_ref': report_output['ref'], 'predictions': predictions_mapping}

        #END predict_phenotype

        # At some point might do deeper type checking...
        if not isinstance(output, dict):
            raise ValueError('Method predict_phenotype return value ' +
                             'output is not type dict as required.')
        # return the results
        # return [output]
        return

    def status(self, ctx):
        #BEGIN_STATUS
        returnVal = {'state': "OK",
                     'message': "",
                     'version': self.VERSION,
                     'git_url': self.GIT_URL,
                     'git_commit_hash': self.GIT_COMMIT_HASH}
        #END_STATUS
        return [returnVal]
