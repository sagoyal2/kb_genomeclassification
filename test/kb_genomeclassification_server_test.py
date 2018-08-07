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

    # def test_download_shock(self):

    """
    def test_workspace(self):
        import biokbase.workspace.client

        current_ws = 'my_current_ws'
        ws = biokbase.narrative.clients.get("workspace")
        ws_client = Workspace()
    """

    """
    def test_build_classifier(self):
        params = {
        "shock_id": "2dcb9741-df3b-4a3a-8ce4-d8dead1b4127", #2dcb9741-df3b-4a3a-8ce4-d8dead1b4127 #72e5c2c1-217d-462f-a0a1-7fd2c1a59f5c
        "list_name": "262543.4,216432.3,269798.12,309807.19,411154.5,485917.5,485918.5,457391.3,470145.6,665954.3,679190.3",
        "phenotypeclass": "Gram_Stain", #you can name this whatever it doesn't matter
        "classifier": "DecisionTreeClassifier",#run_all DecisionTreeClassifier LogisticRegression
        "attribute": "functional_roles",
        "save_ts": 0,
        "classifier_out": "GramOut",
        "workspace" : "janakakbase:narrative_1533153056355" #"janakakbase:narrative_1533153056355" "janakakbase:narrative_1533320423326"
        }

        self.getImpl().build_classifier(self.getContext(), params)
    """

    def test_predict_phenotype(self):
        params = {
        "shock_id" : "fill in later",
        "classifier_name" : "GramOut",
        "workspace" : "janakakbase:narrative_1533153056355" #"janakakbase:narrative_1533153056355" "janakakbase:narrative_1533320423326"
        }

        self.getImpl().predict_phenotype(self.getContext(), params)

    """
    def test_build_classifier(self):
        #result = self.getImpl() #impl_kb_genomeclassification = kb_genomeclassification(u"Metabolism")
        print("this first part is about to begin")

        # result = 

        print("this first part is done")

        wsName = self.getWsName()

        print(wsName)

        params = {'phenotypeclass': 'Gram_Stain', #Metabolism Gram_Stain
                  'classifier': 'KNeighborsClassifier',
                  'input_ws': wsName} #run_all KNeighborsClassifier

        print("here is wsName:")
        print(params.get('input_ws'))

        # params = {'target': 'Metabolism'}
        #print(result.build_classifier(self.getContext(), {"gram negative", "xyz"})) #impl_kb_genomeclassification.build_classifier(u"Metabolism")
        

        #self.serviceImpl.build_classifier(self.getContext(), params)

        #cls.cfg[u'which_target'] = u"Metabolism"

        self.getImpl().build_classifier(self.getContext(), params)

        # kb_genomeclassification("Gram_Stain").build_classifier(self.getContext(), params)

        print("done with test_build_classifier")

        #result.predict_phenotype(self.getContext(), {"unknown annotations"})
        #print("done with predict_phenotype")

        #print(result.tree_code()) # <-- prints the tree to see the "decisions" that the classifier is making)
        #print("done with tree_code")


        
        test all classifiers so user can see results
        (however you will also import these classification types)
        result.classiferTest(KNeighborsClassifier(),"Metabolism_KNeighborsClassifier",True)
        result.classiferTest(GaussianNB(),"Metabolism_GaussianNB",True)
        result.classiferTest(LogisticRegression(random_state=0),"Metabolism_LogisticRegression",True)
        result.classiferTest(DecisionTreeClassifier(random_state=0),"Metabolism_DecisionTreeClassifier",True)
        result.classiferTest(svm.LinearSVC(random_state=0),"Metabolism_SVM",True)
        


        #print("done with test_build_classifier")

        #result.build_classifier(self.getContext(), {"gram negative", "xyz"})
        #result.classiferTest(KNeighborsClassifier(),"Metabolism-KNeighborsClassifier",True)

        #impl_kb_genomeclassification.tree_code() # <-- prints the tree to see the "decisions" that the classifier is making
    """
