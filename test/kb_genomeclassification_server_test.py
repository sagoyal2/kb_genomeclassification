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
        """
        params = {        
        "Annotated": 1,
        "shock_id": "",
        "list_name": "Genome_ID	Classification\n262543.4	facultative\n1134785.3	facultative\n269798.12	aerobic\n309807.19	aerobic\n411154.5	aerobic\n485917.5	aerobic\n485918.5	aerobic\n457391.3	anaerobic\n470145.6	anaerobic\n665954.3	anaerobic\n679190.3	anaerobic",
        "description": "myFullGotest",
        "classifier_name": "myFullGoCLF",
        "attribute": "functional_roles",
        "phenotypeclass": "Respiration",
        "workspace" : "sagoyal:narrative_1536939130038"
        }
        """
        
        """
        params = {
        "Annotated": 1,
        "shock_id": "a753ec76-df84-4447-95f5-7186a755fc3b",
        "list_name": "",
        "description": "myFullGoFail",
        "classifier_name": "myFullGoCLF",
        "attribute": "functional_roles",
        "phenotypeclass": "Respiration",
        "workspace" : "sagoyal:narrative_1536939130038"
        }
        """

        # params = {
        # "Annotated": 1,
        # "list_name": "Genome_ID\n204669.6\n234267.13\n240015.3\n1806.1",
        # "description": "Genomes to Predict",
        # "classifier_name": "Respiration_SVM",
        # "attribute": "functional_roles",
        # "phenotypeclass": "Respiration",
        # "workspace" : "janakakbase:narrative_1507002735148"
        # }
        # self.getImpl().predict_phenotype(self.getContext(), params)
 
        pass

    def test_build_classifier(self):
    
        """
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

        """
        params = {
        "save_ts": 1,
        "description": "my Phylum Classifier",
        "trainingset_name": "WorkingPSET",
        "phenotypeclass": "Phylum",
        "classifier": "run_all",
        "attribute": "functional_roles",
        "k_nearest_neighbors": None,
        "gaussian_nb": None,
        "logistic_regression": None,
        "decision_tree_classifier": None,
        "support_vector_machine": None,
        "neural_network": None,
        "ensemble_model": None,
        "classifier_out": "trialPhy",
        "workspace" : "sagoyal:narrative_1534259992668"
        }
        """
        """
        params = {
        "save_ts": 1,
        "description": "my description",
        "trainingset_name": "fromTerminal",
        "phenotypeclass": "phenoterminal",
        "classifier": "run_all",
        "attribute": "functional_roles",
        "k_nearest_neighbors": None,
        "gaussian_nb": None,
        "logistic_regression": None,
        "decision_tree_classifier": None,
        "support_vector_machine": None,
        "neural_network": None,
        "ensemble_model": None,
        "classifier_out": "CLFfromTerminal",
        "workspace" : "sagoyal:narrative_1534292322496"
        }
        """
            
        
        # params =   {
        # "save_ts": 1,
        # "description": "testingAgain",
        # "trainingset_name": "biggerThursday",
        # "phenotypeclass": "myPhenotype",
        # "classifier": "run_all",#"GaussianNB",#"KNeighborsClassifier",
        # "attribute": "functional_roles",
        # "k_nearest_neighbors": None,
        # "gaussian_nb": None,
        # "logistic_regression": None,
        # "decision_tree_classifier": None,
        # "support_vector_machine": None,
        # "neural_network": None,
        # "ensemble_model": None,
        # "classifier_out": "MaysevenGaussianNB",
        # "workspace" : "sagoyal:narrative_1536939130038"
        # }

        # params = {
        # "save_ts": 1,
        # "description": "myFullGoEverything",
        # "trainingset_name": "myFullGo",
        # "phenotypeclass": "Respiration",
        # "classifier": "run_all",
        # "attribute": "functional_roles",
        # "k_nearest_neighbors": None,
        # "gaussian_nb": None,
        # "logistic_regression": None,
        # "decision_tree_classifier": None,
        # "support_vector_machine": None,
        # "neural_network": None,
        # "ensemble_model": None,
        # "classifier_out": "myFullGoEverythingCLF",
        # "workspace" : "sagoyal:narrative_1536939130038"
        # }

        params = {
        "save_ts": 1,
        "description": "mywaydescription",
        "trainingset_name": "myWayTSEt",
        "phenotypeclass": "myway",
        "classifier": "KNeighborsClassifier",
        "attribute": "functional_roles",
        "k_nearest_neighbors": None,
        "gaussian_nb": None,
        "logistic_regression": None,
        "decision_tree_classifier": None,
        "support_vector_machine": None,
        "neural_network": None,
        "ensemble_model": None,
        "classifier_out": "myWayCLF",
        "workspace" : "sagoyal:narrative_1536939130038"
        }
        self.getImpl().build_classifier(self.getContext(), params)
        

    def test_upload_trainingset(self):
        
        """
        params = {
        "shock_id": "1b23efad-fe6d-4e41-b180-37fc6dcb558d",
        "list_name": "Genome_ID Classification\nShewanella_ondeisensis_MR-1_GenBank Aerobic\ngenBankG5O Anaerobic\nNC_003197    Facultative\nGCF_000010525.1    Facultative\nGCF_000007365.1    Aerobic\nGCF_000007725.1    Anaerobic\nGCF_000009605.1  Aerobic\nGCF_000021065.1    Anaerobic\nGCF_000021085.1  Facultative\nGCF_000090965.1    Facultative\nGCF_000174075.1    Aerobic\nGCF_000183225.1    Aerobic\nGCF_000183245.1    Facultative\nGCF_000183285.1    Facultative\nGCF_000183305.1    Anaerobic\nGCF_000217635.1  Aerobic\nGCF_000225445.1    Aerobic\nGCF_000225465.1    Anaerobic\nGCF_000521525.1  Anaerobic\nGCF_000521545.1  Aerobic\nGCF_000521565.1    Aerobic\nGCF_000521585.1    Facultative\nGCF_001280225.1    Anaerobic\nGCF_001648115.1  Facultative\nGCF_001700895.1    Aerobic\nGCF_001939165.1    Anaerobic\nGCF_003099975.1  Facultative\nGCF_900016785.1    Facultative\nGCF_900128595.1    Aerobic\nGCF_900128725.1    Anaerobic\nGCF_900128735.1  Aerobic\nGCF_000218545.1    Anaerobic\nGCF_000020965.1  Facultative\nGCF_000378225.1    Facultative\nGCF_000012885.1    Aerobic\nGCF_001375595.1    Aerobic\nGCF_000518705.1    Facultative\nGCF_001735525.1    Facultative\nGCF_000016585.1    Anaerobic\nGCF_000169215.2  Aerobic\nGCF_000519065.1    Aerobic\nGCF_001591325.1    Anaerobic\nGCF_002157365.1  Facultative\nGCF_003315425.1    Aerobic\nGCF_000219105.1    Aerobic\nGCF_000988565.1    Aerobic\nGCF_900111765.1    Anaerobic\nGCF_000012685.1  Facultative\nGCF_000278585.1    Anaerobic",
        "description": "trial description in terminal",
        "phenotypeclass": "newTest",
        "training_set_out": "fourColumn",
        "workspace" : "sagoyal:narrative_1534292322496"
        }
        """
        """
        params = {
        "shock_id": "502d096b-4236-462b-addf-9b7b56ff7b64",#"c2203dc8-01db-45a7-a246-09e9fade7d7a",
        "list_name": "",
        "description": "rast testing",
        "phenotypeclass": "RASTPheno",
        "training_set_out": "rastOut",
        "workspace" : "sagoyal:narrative_1536939130038" #"sagoyal:narrative_1534292322496"
        }
        """
        
        """
        params = {
        "Annotated": 0,
        # "shock_id": "",
        "Upload_File" : "/kb/module/data/testingData/prodTrial.xlsx",
        # "list_name": "Genome_ID Classification\n679190.3.RAST   facultative\n665954.3.RAST  facultative\n470145.6.RAST  aerobic\n457391.3.RAST  aerobic\n485918.5.RAST  anaerobic\n411154.5.RAST    anaerobic\n309807.19.RAST   facultative\n269798.12.RAST aerobic\n216432.3.RAST  anaerobic",
        "description": "myRASTtest",
        "phenotypeclass": "respiration",
        "training_set_out": "FullGo",
        "workspace" : "sagoyal:narrative_1536939130038"
        }
        """
        """
        params = {
        "Annotated": 1,
        "Upload_File": "prodTrialRAST.xlsx",
        "list_name": "Genome_ID	Classification\n262543.4.RAST	facultative\n1134785.3.RAST	facultative\n216432.3.RAST	aerobic\n269798.12.RAST	aerobic\n309807.19.RAST	aerobic\n411154.5.RAST	aerobic\n485917.5.RAST	aerobic\n485918.5.RAST	aerobic\n457391.3.RAST	anaerobic\n470145.6.RAST	anaerobic\n665954.3.RAST	anaerobic\n679190.3.RAST	anaerobic",
        "description": "myStagingTrial",
        "phenotypeclass": "Respiration",
        "training_set_out": "StagingRespiration",
        "workspace" : "sagoyal:narrative_1536939130038"
        }

        self.getImpl().upload_trainingset(self.getContext(), params)
        """
        pass
