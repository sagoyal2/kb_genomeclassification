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
    GIT_URL = "https://github.com/sagoyal2/kb_genomeclassification.git"
    GIT_COMMIT_HASH = "d7f43fbbea9bf1d87fec3132a97f42a9b70d10c9"

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
        pass


    def build_classifier(self, ctx, params):
        """
        build_classifier: build_classifier
        requried params:
        :param params: instance of type "BuildClassifierInput" -> structure:
           parameter "genome_attribute" of String, parameter "workspace" of
           String, parameter "training_set_name" of String, parameter
           "classifier_training_set" of mapping from String to type
           "ClassifierTrainingSet" -> structure: parameter "phenotype" of
           String, parameter "genome_name" of String, parameter
           "classifier_object_name" of String, parameter "description" of
           String, parameter "classifier_to_run" of String, parameter
           "logistic_regression" of type "LogisticRegressionOptions" ->
           structure: parameter "penalty" of String, parameter "dual" of type
           "boolean" ("True" or "False"), parameter "lr_tolerance" of Double,
           parameter "lr_C" of Double, parameter "fit_intercept" of type
           "boolean" ("True" or "False"), parameter "intercept_scaling" of
           Double, parameter "lr_class_weight" of String, parameter
           "lr_random_state" of Long, parameter "lr_solver" of String,
           parameter "lr_max_iter" of Long, parameter "multi_class" of
           String, parameter "lr_verbose" of type "boolean" ("True" or
           "False"), parameter "lr_warm_start" of Long, parameter "lr_n_jobs"
           of Long, parameter "decision_tree_classifier" of type
           "DecisionTreeClassifierOptions" -> structure: parameter
           "criterion" of String, parameter "splitter" of String, parameter
           "max_depth" of Long, parameter "min_samples_split" of Long,
           parameter "min_samples_leaf" of Long, parameter
           "min_weight_fraction_leaf" of Double, parameter "max_features" of
           String, parameter "dt_random_state" of Long, parameter
           "max_leaf_nodes" of Long, parameter "min_impurity_decrease" of
           Double, parameter "dt_class_weight" of String, parameter "presort"
           of String, parameter "gaussian_nb" of type "GaussianNBOptions" ->
           structure: parameter "priors" of String, parameter
           "k_nearest_neighbors" of type "KNearestNeighborsOptions" ->
           structure: parameter "n_neighbors" of Long, parameter "weights" of
           String, parameter "algorithm" of String, parameter "leaf_size" of
           Long, parameter "p" of Long, parameter "metric" of String,
           parameter "metric_params" of String, parameter "knn_n_jobs" of
           Long, parameter "support_vector_machine" of type
           "SupportVectorMachineOptions" -> structure: parameter "svm_C" of
           Double, parameter "kernel" of String, parameter "degree" of Long,
           parameter "gamma" of String, parameter "coef0" of Double,
           parameter "probability" of type "boolean" ("True" or "False"),
           parameter "shrinking" of type "boolean" ("True" or "False"),
           parameter "svm_tolerance" of Double, parameter "cache_size" of
           Double, parameter "svm_class_weight" of String, parameter
           "svm_verbose" of type "boolean" ("True" or "False"), parameter
           "svm_max_iter" of Long, parameter "decision_function_shape" of
           String, parameter "svm_random_state" of Long, parameter
           "neural_network" of type "NeuralNetworkOptions" -> structure:
           parameter "hidden_layer_sizes" of String, parameter "activation"
           of String, parameter "mlp_solver" of String, parameter "alpha" of
           Double, parameter "batch_size" of String, parameter
           "learning_rate" of String, parameter "learning_rate_init" of
           Double, parameter "power_t" of Double, parameter "mlp_max_iter" of
           Long, parameter "shuffle" of type "boolean" ("True" or "False"),
           parameter "mlp_random_state" of Long, parameter "mlp_tolerance" of
           Double, parameter "mlp_verbose" of type "boolean" ("True" or
           "False"), parameter "mlp_warm_start" of type "boolean" ("True" or
           "False"), parameter "momentum" of Double, parameter
           "nesterovs_momentum" of type "boolean" ("True" or "False"),
           parameter "early_stopping" of type "boolean" ("True" or "False"),
           parameter "validation_fraction" of Double, parameter "beta_1" of
           Double, parameter "beta_2" of Double, parameter "epsilon" of
           Double, parameter "ensemble_model" of type "EnsembleModelOptions"
           -> structure: parameter "k_nearest_neighbors_box" of Long,
           parameter "gaussian_nb_box" of Long, parameter
           "logistic_regression_box" of Long, parameter
           "decision_tree_classifier_box" of Long, parameter
           "support_vector_machine_box" of Long, parameter
           "neural_network_box" of Long, parameter "voting" of String,
           parameter "en_weights" of String, parameter "en_n_jobs" of Long,
           parameter "flatten_transform" of type "boolean" ("True" or "False")
        :returns: instance of type "ClassifierOut" -> structure: parameter
           "classifier_info" of list of type "classifierInfo" -> structure:
           parameter "classifier_name" of String, parameter "classifier_ref"
           of String, parameter "accuracy" of Double, parameter "report_name"
           of String, parameter "report_ref" of String
        """
        # ctx is the context object
        # return variables are: output
        #BEGIN build_classifier

        self.config['ctx'] = ctx
        build_runner = kb_genomeclfUtils(self.config)

        html_output_name, classifier_info_list = build_runner.fullClassify(params, params['workspace'])
        report_output =  build_runner.generateHTMLReport(params['workspace'], "forBuild", html_output_name, params['description'], for_build_classifier=True)
        output = {'report_name': report_output['name'], 'report_ref': report_output['ref'], 'classifier_info': classifier_info_list}
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
           "categorizer_name" of String, parameter "description" of String,
           parameter "file_path" of String, parameter "annotate" of Long
        :returns: instance of type "ClassifierPredictionOutput" -> structure:
           parameter "prediction_set" of mapping from String to type
           "PredictedPhenotypeOut" -> structure: parameter
           "prediction_probabilities" of Double, parameter "phenotype" of
           String, parameter "genome_name" of String, parameter "genome_ref"
           of String, parameter "report_name" of String, parameter
           "report_ref" of String
        """
        # ctx is the context object
        # return variables are: output
        #BEGIN predict_phenotype
        self.config['ctx'] = ctx
        predict_Runner = kb_genomeclfUtils(self.config)

        html_output_name, prediction_set = predict_Runner.fullPredict(params, params['workspace'])
        report_output = predict_Runner.generateHTMLReport(params['workspace'], "forPredict", html_output_name, params['description'])
        output = {'report_name': report_output['name'], 'report_ref': report_output['ref'], 'prediction_set': prediction_set}

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
           structure: parameter "phenotype" of String, parameter "workspace"
           of String, parameter "workspace_id" of String, parameter
           "description" of String, parameter "training_set_name" of String,
           parameter "file_path" of String, parameter "annotate" of Long
        :returns: instance of type "UploadTrainingSetOut" -> structure:
           parameter "classifier_training_set" of mapping from String to type
           "ClassifierTrainingSetOut" -> structure: parameter "phenotype" of
           String, parameter "genome_name" of String, parameter "genome_ref"
           of String, parameter "references" of list of String, parameter
           "evidence_types" of list of String, parameter "report_name" of
           String, parameter "report_ref" of String
        """
        # ctx is the context object
        # return variables are: output
        #BEGIN upload_trainingset

        self.config['ctx'] = ctx
        upload_runner = kb_genomeclfUtils(self.config)

        html_output_name, classifier_training_set = upload_runner.fullUpload(params, params['workspace'])
        report_output = upload_runner.generateHTMLReport(params['workspace'], "forUpload", html_output_name, params['description'])
        output = {'report_name': report_output['name'], 'report_ref': report_output['ref'], 'classifier_training_set': classifier_training_set}

        #END upload_trainingset

        # At some point might do deeper type checking...
        if not isinstance(output, dict):
            raise ValueError('Method upload_trainingset return value ' +
                             'output is not type dict as required.')
        # return the results
        return [output]

    def rast_annotate_trainingset(self, ctx, params):
        """
        :param params: instance of type "RastAnnotateTrainingSetInput" ->
           structure: parameter "classifier_training_set" of mapping from
           String to type "ClassifierTrainingSetOut" -> structure: parameter
           "phenotype" of String, parameter "genome_name" of String,
           parameter "genome_ref" of String, parameter "references" of list
           of String, parameter "evidence_types" of list of String, parameter
           "workspace" of String, parameter "make_genome_set" of Long
        :returns: instance of type "RastAnnotateTrainingSetOutput" ->
           structure: parameter "classifier_training_set" of mapping from
           String to type "ClassifierTrainingSetOut" -> structure: parameter
           "phenotype" of String, parameter "genome_name" of String,
           parameter "genome_ref" of String, parameter "references" of list
           of String, parameter "evidence_types" of list of String, parameter
           "report_name" of String, parameter "report_ref" of String
        """
        # ctx is the context object
        # return variables are: output
        #BEGIN rast_annotate_trainingset

        self.config['ctx'] = ctx
        annotate_runner = kb_genomeclfUtils(self.config)

        html_output_name, classifier_training_set= annotate_runner.fullAnnotate(params, params['workspace'])
        report_output = annotate_runner.generateHTMLReport(params['workspace'], "forAnnotate", html_output_name, params['description'])
        output = {'report_name': report_output['name'], 'report_ref': report_output['ref'], 'classifier_training_set': classifier_training_set}

        #END rast_annotate_trainingset

        # At some point might do deeper type checking...
        if not isinstance(output, dict):
            raise ValueError('Method rast_annotate_trainingset return value ' +
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









