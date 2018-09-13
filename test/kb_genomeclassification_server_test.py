# -*- coding: utf-8 -*-
import unittest
import os  # noqa: F401
import json  # noqa: F401
import time
import requests

from os import environ


try:
    from ConfigParser import ConfigParser  # py2
except:
    from configparser import ConfigParser  # py3


from pprint import pprint  # noqa: F401

from biokbase.workspace.client import Workspace as workspaceService
from kb_genomeclassification.kb_genomeclassificationImpl import kb_genomeclassification
from kb_genomeclassification.kb_genomeclassificationServer import MethodContext
from kb_genomeclassification.authclient import KBaseAuth as _KBaseAuth

from AssemblyUtil.AssemblyUtilClient import AssemblyUtil

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

    def test_predict_phenotype(self):
        pass

    def test_build_classifier(self):
    
        params = {
        "save_ts": 1,
        "description": "Respiration Classifier",
        "trainingset_name": "TrainingRespiration",
        "phenotypeclass": "Respiration",
        "classifier": "run_all",
        "attribute": "functional_roles",
        "k_nearest_neighbors": None,
        "gaussian_nb": None,
        "logistic_regression": None,
        "decision_tree_classifier": None,
        "support_vector_machine": None,
        "neural_network": None,
        "ensemble_model": None,
        "classifier_out": "myRCLF",
        "workspace" : "sagoyal:narrative_1534259992668"
        }

        """
        params = {
        "save_ts": 1,
        "description": "Gram Classifier",
        "trainingset_name": "TrainingGram",
        "phenotypeclass": "Gram",
        "classifier": "run_all",
        "attribute": "functional_roles",
        "k_nearest_neighbors": None,
        "gaussian_nb": None,
        "logistic_regression": None,
        "decision_tree_classifier": None,
        "support_vector_machine": None,
        "neural_network": None,
        "ensemble_model": None,
        "classifier_out": "myGCLF",
        "workspace" : "sagoyal:narrative_1534259992668"
        }
        """

        self.getImpl().build_classifier(self.getContext(), params)

    def test_upload_trainingset(self):
        pass
