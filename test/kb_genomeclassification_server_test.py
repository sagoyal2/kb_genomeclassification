# -*- coding: utf-8 -*-
import unittest
import os  # noqa: F401
import json  # noqa: F401
import time
import requests
import uuid
import pickle

from os import environ


try:
    from configparser import ConfigParser  # py2 
except:
    from configparser import ConfigParser  # py3


from pprint import pprint 

from biokbase.workspace.client import Workspace as workspaceService
from kb_genomeclassification.kb_genomeclassificationImpl import kb_genomeclassification
from kb_genomeclassification.kb_genomeclassificationServer import MethodContext
from kb_genomeclassification.authclient import KBaseAuth as _KBaseAuth

from AssemblyUtil.AssemblyUtilClient import AssemblyUtil
from DataFileUtil.DataFileUtilClient import DataFileUtil

class kb_genomeclassificationTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        token = environ.get('KB_AUTH_TOKEN', None)
        config_file = environ.get('KB_DEPLOYMENT_CONFIG', None)
        cls.cfg = {}
        config = ConfigParser()
        config.read(config_file)
        for nameval in config.items('kb_genomeclassification'):
            cls.cfg[nameval[0]] = nameval[1]
        # Getting username from Auth profile for token
        authServiceUrl = cls.cfg['auth-service-url']
        auth_client = _KBaseAuth(authServiceUrl)
        user_id = auth_client.get_user(token)
        # WARNING: don't call any logging methods on the context object,
        # it'll result in a NoneType error
        cls.ctx = MethodContext(None)
        cls.ctx.update({'token': token,
                        'user_id': user_id,
                        'provenance': [
                            {'service': 'kb_genomeclassification',
                             'method': 'please_never_use_it_in_production',
                             'method_params': []
                             }],
                        'authenticated': 1})
        cls.wsURL = cls.cfg['workspace-url']
        cls.wsClient = workspaceService(cls.wsURL)
        cls.serviceImpl = kb_genomeclassification(cls.cfg)
        cls.scratch = cls.cfg['scratch']
        cls.callback_url = os.environ['SDK_CALLBACK_URL']

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, 'wsName'):
            cls.wsClient.delete_workspace({'workspace': cls.wsName})
            print('Test workspace was deleted')

    def getWsClient(self):
        return self.__class__.wsClient

    def getWsName(self):
        if hasattr(self.__class__, 'wsName'):
            return self.__class__.wsName
        suffix = int(time.time() * 1000)
        wsName = "test_kb_genomeclassification_" + str(suffix)
        ret = self.getWsClient().create_workspace({'workspace': wsName})  # noqa
        self.__class__.wsName = wsName
        return wsName

    def getImpl(self):
        return self.__class__.serviceImpl

    def getContext(self):
        return self.__class__.ctx

    def test_upload_trainingset(self):
        pass
        # params = {
        # "annotate": 0,
        # "file_path": "fake_2_refseq.xlsx",
        # "description": "my description",
        # "phenotype": "my phenotype",
        # "training_set_name": "AgainRefSeq",
        # "workspace": "sagoyal:narrative_1536939130038",
        # "workspace_id":"36230"
        # }
        # self.getImpl().upload_trainingset(self.getContext(), params)

        # params =    {
        # "annotate": 0,
        # "file_path": "full_genomeid_classification.xlsx",
        # "description": "full test 1",
        # "phenotype": "Respiration",
        # "training_set_name": "RespirationTrainingSet",
        # "workspace": "sagoyal:narrative_1536939130038"
        # }
        # self.getImpl().upload_trainingset(self.getContext(), params)

    def test_rast_annotate_trainingset(self):

        params =    {
        "training_set_name": "AgainRefSeq",
        "description": "whatever",
        "annotated_trainingset_name": "AnnoatedAgainRefSeq",
        "workspace": "sagoyal:narrative_1536939130038"
        }
        self.getImpl().rast_annotate_trainingset(self.getContext(), params)

    def test_build_classifier(self):
        pass
        # params = {
        # "description": "my build classifier description",
        # "training_set_name": "to_try_with_build",
        # "classifier_to_run": "run_all",
        # "genome_attribute": "functional_roles",
        # "k_nearest_neighbors": None,
        # "gaussian_nb": None,
        # "logistic_regression": None,
        # "decision_tree_classifier": None,
        # "support_vector_machine": None,
        # "neural_network": None,
        # "ensemble_model": None,
        # "classifier_object_name": "clf_name",
        # "workspace": "sagoyal:narrative_1536939130038"
        # }
        # params = {
        # "description": "my build classifier description",
        # "training_set_name": "to_try_with_build",
        # "classifier_to_run": "run_all",
        # "genome_attribute": "functional_roles",
        # "k_nearest_neighbors": {
        #     "n_neighbors": 5,
        #     "weights": "uniform",
        #     "algorithm": "auto",
        #     "leaf_size": 30,
        #     "p": 2,
        #     "metric": "minkowski",
        # },
        # "gaussian_nb": {
        #     "priors": "None"
        # },
        # "logistic_regression": {
        #     "penalty": "l2",
        #     "dual": "False",
        #     "lr_tolerance": 0.0001,
        #     "lr_C": 1,
        #     "fit_intercept": "True",
        #     "intercept_scaling": 1,
        #     "lr_solver": "newton-cg",
        #     "lr_max_iter": 100,
        #     "multi_class": "ovr",
        # },
        # "decision_tree_classifier": {
        #     "criterion": "gini",
        #     "splitter": "best",
        #     "max_depth": None,
        #     "min_samples_split": 2,
        #     "min_samples_leaf": 1,
        #     "min_weight_fraction_leaf": 0,
        #     "max_leaf_nodes": None,
        #     "min_impurity_decrease": 0
        # },
        # "support_vector_machine": {
        #     "svm_C": 1,
        #     "kernel": "linear",
        #     "degree": 3,
        #     "gamma": "auto",
        #     "coef0": 0,
        #     "probability": "False",
        #     "shrinking": "True",
        #     "svm_tolerance": 0.001,
        #     "cache_size": 200,
        #     "svm_max_iter": -1,
        #     "decision_function_shape": "ovr"
        # },
        # "neural_network": {
        #     "hidden_layer_sizes": "100",
        #     "activation": "relu",
        #     "mlp_solver": "adam",
        #     "alpha": 0.0001,
        #     "batch_size": "auto",
        #     "learning_rate": "constant",
        #     "learning_rate_init": 0.001,
        #     "power_t": 0.05,
        #     "mlp_max_iter": 200,
        #     "shuffle": "True",
        #     "mlp_random_state": 0,
        #     "mlp_tolerance": 0.0001,
        #     "mlp_verbose": "False",
        #     "mlp_warm_start": "False",
        #     "momentum": 0.9,
        #     "nesterovs_momentum": "True",
        #     "early_stopping": "False",
        #     "validation_fraction": 0.1,
        #     "beta_1": 0.9,
        #     "beta_2": 0.999,
        #     "epsilon": 1e-8
        # },
        # "classifier_object_name": "clf_name",
        # "workspace": "sagoyal:narrative_1536939130038"
        # }
        # self.getImpl().build_classifier(self.getContext(), params)

        # params = {
        # "description": "Tree Figure Testing",
        # "training_set_name": "RespirationTrainingSet",
        # "classifier_to_run": "decision_tree_classifier",
        # "genome_attribute": "functional_roles",
        # "k_nearest_neighbors": None,
        # "gaussian_nb": None,
        # "logistic_regression": None,
        # "decision_tree_classifier": None,
        # "support_vector_machine": None,
        # "neural_network": None,
        # "classifier_object_name": "TreeFigureTest",
        # "workspace": "sagoyal:narrative_1536939130038"
        # }
        # self.getImpl().build_classifier(self.getContext(), params)

    def test_predict_phenotype(self):
        pass
        # params = {
        # "categorizer_name": "clf_name_k_nearest_neighbors",
        # "annotate": 0,
        # "file_path": "GramDataEdit5.xlsx",
        # "description": "my predict phenotype description",
        # "workspace": "sagoyal:narrative_1536939130038",
        # "workspace_id":"36230"
        # }
        # self.getImpl().predict_phenotype(self.getContext(), params)
























