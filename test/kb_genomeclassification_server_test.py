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
   
    def test_build_classifier(self):
        """
        params = {
        "shock_id": "5159651e-fdc1-4d2f-a7d3-d79948d2c9e7",#"2dcb9741-df3b-4a3a-8ce4-d8dead1b4127", #2dcb9741-df3b-4a3a-8ce4-d8dead1b4127 #72e5c2c1-217d-462f-a0a1-7fd2c1a59f5c
        #"list_name": "Genome_ID Classification\nShewanella_ondeisensis_MR-1_GenBank A\ngenBankGO    B\nNC_003197    C\nGCF_000010525.1  B\nGCF_000007365.1  B\nGCF_000007725.1  C\nGCF_000009605.1  A\nGCF_000021065.1  C\nGCF_000021085.1  C\nGCF_000090965.1  A\nGCF_000174075.1  B\nGCF_000183225.1  C\nGCF_000183245.1  B\nGCF_000183285.1  B\nGCF_000183305.1  C\nGCF_000217635.1  A\nGCF_000225445.1  C\nGCF_000225465.1  C\nGCF_000521525.1  A\nGCF_000521545.1  B\nGCF_000521565.1  C\nGCF_000521585.1  B\nGCF_001280225.1  B\nGCF_001648115.1  C\nGCF_001700895.1  A\nGCF_001939165.1  C\nGCF_003099975.1  C\nGCF_900016785.1  A\nGCF_900128595.1  B\nGCF_900128725.1  C\nGCF_900128735.1  B\nGCF_000218545.1  B\nGCF_000020965.1  C\nGCF_000378225.1  A\nGCF_000012885.1  C\nGCF_001375595.1  C\nGCF_000518705.1  A\nGCF_001735525.1  B\nGCF_000016585.1  C\nGCF_000169215.2  B\nGCF_000519065.1  B\nGCF_001591325.1  C\nGCF_002157365.1  A\nGCF_003315425.1  C\nGCF_000219105.1  C\nGCF_000988565.1  A\nGCF_900111765.1  B\nGCF_000012685.1  C\nGCF_000278585.1  B\nGCF_000340515.1  B\nGCF_900106535.1  C\nGCF_002305895.1  A\nGCF_000165485.1  C\nGCF_000168055.1  C\nGCF_900109545.1  A\nGCF_000335475.2  B\nGCF_002305875.1  C\nGCF_001027285.1  B\nGCF_000601485.1  B\nGCF_001189295.1  C\nGCF_002343915.1  A\nGCF_900112715.1  C\nGCF_000067165.1  C\nGCF_000418325.1  A\nGCF_001589285.1  B\nGCF_002950945.1  C\nGCF_000382305.1  B\nGCF_002222655.1  B\nGCF_001442515.1  C\nGCF_002024615.1  A\nGCF_002355975.1  C\nGCF_900106525.1  C\nGCF_000163775.2  A\nGCF_002951835.1  B\nGCF_000685215.1  C\nGCF_002954665.1  B\nGCF_000427405.1  B\nGCF_000222485.1  C\nGCF_000218895.1  A\nGCF_000171775.1  C\nGCF_003149495.1  C\nGCF_000165715.2  A\nGCF_000092105.1  B\nGCF_000181475.1  C\nGCF_000153105.1  B\nGCF_002967715.1  B\nGCF_002967725.1  C\nGCF_002967755.1  A\nGCF_002967765.1  C\nGCF_003044115.1  C\nGCF_000025185.1  A\nGCF_000186345.1  B\nGCF_000283235.1  C\nGCF_000008685.2  B\nGCF_000021405.1  B\nGCF_000166635.1  C\nGCF_000166655.1  A\nGCF_000171735.2  C\nGCF_000171755.2  C\nGCF_000172295.2  A\nGCF_000172315.2  B\nGCF_000172335.2  C\nGCF_000181555.2  B\nGCF_000181575.2  B\nGCF_000181715.2  C\nGCF_000181855.2  A\nGCF_000382565.1  C\nGCF_000444465.1  C\nGCF_000501295.1  A\nGCF_000501315.1  B\nGCF_000501335.1  C\nGCF_000501375.1  B\nGCF_000501455.1  B\nGCF_000501535.1  C\nGCF_000502035.1  A\nGCF_000502055.1  C\nGCF_000502075.1  C\nGCF_000502135.1  A\nGCF_000502155.1  B\nGCF_000769595.1  C\nGCF_002083055.2  B\nGCF_002151465.1  B\nGCF_002151485.1  C\nGCF_002151505.1  A\nGCF_002442595.1  C\nGCF_002995865.1  C\nGCF_002995885.1  A\nGCF_002995905.1  B\nGCF_002995915.1  C\nGCF_002995935.1  B\nGCF_002995965.1  B\nGCF_002995985.1  C\nGCF_002996005.1  A\nGCF_002996025.1  C\nGCF_002996045.1  C\nGCF_002996065.1  A\nGCF_002996085.1  B\nGCF_002996105.1  C\nGCF_002996125.1  B\nGCF_002996145.1  B\nGCF_002996165.1  C\nGCF_002996175.1  A\nGCF_002996185.1  C\nGCF_002996225.1  C\nGCF_002996245.1  A\nGCF_002996265.1  B\nGCF_002996275.1  C\nGCF_002996285.1  B\nGCF_002996305.1  B\nGCF_002996345.1  C\nGCF_002996365.1  A\nGCF_002996385.1  C\nGCF_002996395.1  C\nGCF_002996405.1  A\nGCF_000795905.1  B\nGCF_002996445.1  C\nGCF_000795925.1  B\nGCF_000589375.1  B\nGCF_002996465.1  C\nGCF_000795945.1  A\nGCF_002996475.1  C\nGCF_000795955.1  C\nGCF_000589395.1  A\nGCF_002996485.1  B\nGCF_000795985.1  C\nGCF_002996525.1  B\nGCF_000589415.1  B\nGCF_000796005.1  C\nGCF_002996545.1  A\nGCF_000796025.1  C\nGCF_000589435.1  C\nGCF_002996565.1  A\nGCF_000796045.1  B\nGCF_002996585.1  C\nGCF_000589455.1  B\nGCF_000796065.1  B\nGCF_002996605.1  C\nGCF_000796085.1  A\nGCF_000589475.1  C\nGCF_002996625.1  C\nGCF_000796095.1  A\nGCF_002996645.1  B\nGCF_000589495.1  C\nGCF_000796125.1  B\nGCF_002996685.1  B\nGCF_000589515.1  C\nGCF_000796145.1  A\nGCF_002996705.1  C\nGCF_002996715.1  C\nGCF_000589535.1  A\nGCF_002996745.1  B\nGCF_000589555.1  C\nGCF_002996785.1  B\nGCF_002996795.1  B\nGCF_000589575.1  C\nGCF_000796175.1  A\nGCF_002996845.1  C\nGCF_000796205.1  C\nGCF_000589595.1  A\nGCF_002996865.1  B\nGCF_000796225.1  C\nGCF_002996895.1  B\nGCF_000589615.1  B\nGCF_002996925.1  C\nGCF_000796245.1  A\nGCF_000589635.1  C\nGCF_002996965.1  C\nGCF_000796255.1  A\nGCF_002996985.1  B\nGCF_000589655.1  C\nGCF_000796285.1  B\nGCF_002997025.1  B\nGCF_000589675.1  C\nGCF_000796295.1  A\nGCF_002997045.1  C\nGCF_000796325.1  C\nGCF_002997065.1  A\nGCF_000589695.1  B\nGCF_000796345.1  C\nGCF_002997085.1  B\nGCF_000589715.1  B\nGCF_000796365.1  C\nGCF_002997105.1  A\nGCF_000796385.1  C\nGCF_002997145.1  C\nGCF_000589735.1  A\nGCF_000796405.1  B\nGCF_002997165.1  C\nGCF_000589755.1  B\nGCF_000796425.1  B\nGCF_002997185.1  C\nGCF_002997205.1  A\nGCF_000796445.1  C\nGCF_000589775.1  C\nGCF_002997245.1  A\nGCF_000796465.1  B\nGCF_000589795.1  C\nGCF_002997265.1  B\nGCF_000796475.1  B\nGCF_002997285.1  C\nGCF_000589815.1  A\nGCF_000796505.1  C\nGCF_000012065.2  C\nGCF_000796525.1  A\nGCF_000589835.1  B\nGCF_000568675.1  C\nGCF_000796545.1  B\nGCF_000568775.1  B\nGCF_000591575.1  C\nGCF_000796555.1  A\nGCF_000568795.1  C\nGCF_000796585.1  C\nGCF_000591595.1  A\nGCF_000956315.1  B\nGCF_000796605.1  C\nGCF_001598195.1  B\nGCF_000591615.1  B\nGCF_000796615.1  C\nGCF_001660005.1  A\nGCF_000591635.1  C\nGCF_000512145.1  C\nGCF_000796645.1  A\nGCF_000568735.2  B\nGCF_000591655.1  C\nGCF_000796655.1  B\nGCF_000012085.2  B\nGCF_000796685.1  C\nGCF_000591675.1  A\nGCF_000568655.1  C\nGCF_000796705.1  C\nGCF_001936255.1  A\nGCF_000591695.1  B\nGCF_000796725.1  C\nGCF_000518125.1  B\nGCF_000796745.1  B\nGCF_000591715.1  C\nGCF_000568755.1  A\nGCF_000796765.1  C\nGCF_000378205.1  C\nGCF_000591735.1  A\nGCF_000796785.1  B\nGCF_000147075.1  C\nGCF_000796795.1  B\nGCF_000591755.1  B\nGCF_000184345.1  C\nGCF_000796825.1  A\nGCF_000008185.1  C\nGCF_000591775.1  C\nGCF_000796845.1  A\nGCF_000338455.1  B\nGCF_000591795.1  C\nGCF_000338475.1  B\nGCF_000796865.1  B\nGCF_000338515.1  C\nGCF_000591815.1  A\nGCF_000338595.1  C\nGCF_000591835.1  C\nGCF_000338615.1  A\nGCF_000338635.1  B\nGCF_000591855.1  C\nGCF_000340605.1  B\nGCF_000591875.1  B\nGCF_000340645.1  C\nGCF_000340685.1  A\nGCF_000591895.1  C\nGCF_000340705.1  C\nGCF_000591915.1  A\nGCF_000340725.1  B\nGCF_000340745.1  C\nGCF_000591935.1  B\nGCF_000413075.1  B\nGCF_000591955.1  C\nGCF_000413095.1  A\nGCF_000413115.1  C\nGCF_000591975.1  C\nGCF_000022105.1  A\nGCF_000591995.1  B\nGCF_000383255.1  C\nGCF_001012915.1  B\nGCF_000592015.1  B\nGCF_001012925.1  C\nGCF_000592035.1  A\nGCF_001012935.1  C\nGCF_001012945.1  C\nGCF_000592055.1  A\nGCF_001012995.1  B\nGCF_000592075.1  C\nGCF_001013005.1  B\nGCF_001013035.1  B\nGCF_000592095.1  C\nGCF_001013045.1  A\nGCF_000592115.1  C\nGCF_001013075.1  C\nGCF_001013085.1  A\nGCF_000592135.1  B\nGCF_001013115.1  C\nGCF_000592155.1  B\nGCF_001013125.1  B\nGCF_001013155.1  C\nGCF_000592175.1  A\nGCF_001013165.1  C\nGCF_000592195.1  C\nGCF_001013195.1  A\nGCF_001013205.1  B\nGCF_000592215.1  C\nGCF_001013235.1  B\nGCF_000592235.1  B\nGCF_001013245.1  C\nGCF_001013275.1  A\nGCF_000592255.1  C\nGCF_001676785.2  C\nGCF_000592275.1  A\nGCF_002100745.1  B\nGCF_002117505.1  C\nGCF_000592295.1  B\nGCF_002117515.1  B\nGCF_000592315.1  C\nGCF_002850235.1  A\nGCF_000024485.1  C\nGCF_000592335.1  C\nGCF_000246755.1  A\nGCF_000592355.1  B\nGCF_000246775.1  C\nGCF_000246795.1  B\nGCF_000592375.1  B\nGCF_000246815.1  C\nGCF_000592395.1  A\nGCF_000304295.1  C\nGCF_000387485.1  C\nGCF_000592415.1  A\nGCF_000410535.2  B\nGCF_000592435.1  C\nGCF_000410555.1  B\nGCF_000604125.1  B\nGCF_000592455.1  C\nGCF_000813285.1  A\nGCF_001628695.1  C\nGCF_001655235.1  C\nGCF_001655275.1  A\nGCF_001655315.1  B\nGCF_001655355.1  C\nGCF_001655395.1  B\nGCF_000592495.1  B\nGCF_001655435.1  C\nGCF_000592515.1  A\nGCF_001655475.1  C\nGCF_001655515.1  C\nGCF_000592535.1  A\nGCF_001655555.1  B\nGCF_000592555.1  C\nGCF_001712895.1  B\nGCF_001712915.1  B\nGCF_000592575.1  C\nGCF_001712935.1  A\nGCF_000592595.1  C\nGCF_001712955.1  C\nGCF_001712975.1  A\nGCF_000592615.1  B\nGCF_001712995.1  C\nGCF_000592635.1  B\nGCF_001713015.1  B\nGCF_001713035.1  C\nGCF_000592655.1  A\nGCF_001713055.1  C\nGCF_000592675.1  C\nGCF_001713075.1  A\nGCF_001713095.1  B\nGCF_000592695.1  C\nGCF_001713115.1  B\nGCF_000592715.1  B\nGCF_001713135.1  C\nGCF_001713155.1  A\nGCF_000592735.1  C\nGCF_000592755.1  C\nGCF_000600375.1  A\nGCF_000600415.1  B\nGCF_000600435.1  C\nGCF_000600455.1  B\nGCF_000600475.1  B\nGCF_000600495.1  C\nGCF_000600515.1  A\nGCF_000600535.1  C\nGCF_000600555.1  C\nGCF_000600575.1  A\nGCF_000600595.1  B\nGCF_000600615.1  C\nGCF_001713195.1  B\nGCF_000600635.1  B\nGCF_001713215.1  C\nGCF_000600655.1  A\nGCF_001713235.1  C\nGCF_001713255.1  C\nGCF_000600675.1  A\nGCF_001713275.1  B\nGCF_000600695.1  C\nGCF_001713295.1  B\nGCF_001713315.1  B\nGCF_000600715.1  C\nGCF_001713335.1  A\nGCF_000600735.1  C\nGCF_001761515.1  C\nGCF_001761565.1  A\nGCF_000600755.1  B\nGCF_001941545.1  C\nGCF_000600775.1  B\nGCF_002271195.1  B\nGCF_002271235.1  C\nGCF_000600795.1  A\nGCF_003055985.1  C\nGCF_000600815.1  C\nGCF_003076575.1  A\nGCF_000187105.1  B\nGCF_000600835.1  C\nGCF_000513775.1  B\nGCF_000600855.1  B\nGCF_000944995.1  C\nGCF_000421345.1  A\nGCF_000600875.1  C\nGCF_900111035.1  C\nGCF_000600895.1  A\nGCF_900113965.1  B\nGCF_000255555.1  C\nGCF_000600915.1  B\nGCF_000195275.1  B\nGCF_000600935.1  C\nGCF_000017605.1  A\nGCF_000017685.1  C\nGCF_000611045.1  C\nGCF_000007685.1  A\nGCF_000611065.1  B\nGCF_000092565.1  C\nGCF_000216035.1  B\nGCF_000611085.1  B\nGCF_000216055.1  C\nGCF_000611105.1  A\nGCF_000216075.1  C\nGCF_000611165.1  C\nGCF_000216095.1  A\nGCF_000216195.1  B\nGCF_000611185.1  C\nGCF_000216235.1  B\nGCF_000611205.1  B\nGCF_000216335.1  C\nGCF_000216355.1  A\nGCF_000611225.1  C\nGCF_000216375.1  C\nGCF_000611245.1  A\nGCF_000216395.1  B\nGCF_000216415.1  C\nGCF_000611265.1  B\nGCF_000216435.1  B\nGCF_000611285.1  C\nGCF_000216475.1  A\nGCF_000216495.1  C\nGCF_000611305.1  C\nGCF_000216515.1  A\nGCF_000611325.1  B\nGCF_000216535.1  C\nGCF_000216555.2  B\nGCF_000611345.1  B\nGCF_000216575.1  C\nGCF_000611365.1  A",
        #"list_name": "Genome_ID Classification\nShewanella_ondeisensis_MR-1_GenBank A\ngenBankGO    B\nNC_003197    C\nGCF_000010525.1  B\nGCF_000007365.1  B\nGCF_000007725.1  C\nGCF_000009605.1  A\nGCF_000021065.1  C\nGCF_000021085.1  C\nGCF_000090965.1  A\nGCF_000174075.1  B\nGCF_000183225.1  C\nGCF_000183245.1  B\nGCF_000183285.1  B\nGCF_000183305.1  C\nGCF_000217635.1  A\nGCF_000225445.1  C\nGCF_000225465.1  C\nGCF_000521525.1  A\nGCF_000521545.1  B\nGCF_000521565.1  C\nGCF_000521585.1  B\nGCF_001280225.1  B\nGCF_001648115.1  C\nGCF_001700895.1  A\nGCF_001939165.1  C\nGCF_003099975.1  C\nGCF_900016785.1  A\nGCF_900128595.1  B\nGCF_900128725.1  C\nGCF_900128735.1  B\nGCF_000218545.1  B\nGCF_000020965.1  C\nGCF_000378225.1  A\nGCF_000012885.1  C\nGCF_001375595.1  C\nGCF_000518705.1  A\nGCF_001735525.1  B\nGCF_000016585.1  C\nGCF_000169215.2  B\nGCF_000519065.1  B\nGCF_001591325.1  C\nGCF_002157365.1  A\nGCF_003315425.1  C\nGCF_000219105.1  C\nGCF_000988565.1  A\nGCF_900111765.1  B\nGCF_000012685.1  C\nGCF_000278585.1  B",
        #"list_name": "Genome_ID Classification\ngenBankGO   N\nNC_003197    P\nShewanella_ondeisensis_MR-1_GenBank  N",#"Genome_ID Classification\ngenBankGO   N\nNC_003197    P",#\nShewanella_ondeisensis_MR-1_GenBank  N",#"",
        "phenotypeclass": "Gram_Stain", #you can name this whatever it doesn't matter
        "classifier": "DecisionTreeClassifier",#run_all DecisionTreeClassifier LogisticRegression
        "attribute": "functional_roles",
        "save_ts": 0,
        "classifier_out": "whyNotDemo",
        "workspace" : "sagoyal:narrative_1534292322496", #"janakakbase:narrative_1533153056355"#"sagoyal:narrative_1533659119242" #"janakakbase:narrative_1533153056355" "janakakbase:narrative_1533320423326"
        }
        
        self.getImpl().build_classifier(self.getContext(), params)
        """
        """
        params = {
        "shock_id" : "8f55c4b4-750f-40c4-a973-f2c98825a18e",
        "list_name" : "Genome_ID\ngenBankGO\nNC_003197",#\nShewanella_ondeisensis_MR-1_GenBank",#"",
        "classifier_name" : "GramOut",
        "phenotypeclass" : "I'm predicting this",
        "workspace" : "sagoyal:narrative_1533659119242" #"janakakbase:narrative_1533153056355" "janakakbase:narrative_1533320423326"
        }

        self.getImpl().predict_phenotype(self.getContext(), params)
        """

    """
    """

    """

    "Genome_ID    Classification\n262543.4    facultative\n1134785.3  facultative\n216432.3   aerobic\n269798.12  aerobic\n309807.19  aerobic\n411154.5   aerobic\n485917.5   aerobic\n485918.5   aerobic\n457391.3   anaerobic\n470145.6 anaerobic\n665954.3 anaerobic\n679190.3 anaerobic"
    "262543.4,216432.3,269798.12,309807.19,411154.5,485917.5,485918.5,457391.3,470145.6,665954.3,679190.3",
    """

    
    def test_predict_phenotype(self):
        """
        params = {
        "shock_id" : "8f55c4b4-750f-40c4-a973-f2c98825a18e",
        "list_name" : "Genome_ID\ngenBankGO\nNC_003197",#\nShewanella_ondeisensis_MR-1_GenBank",#"",
        "classifier_name" : "GramOut",
        "phenotypeclass" : "I'm predicting this",
        "workspace" : "sagoyal:narrative_1533659119242" #"janakakbase:narrative_1533153056355" "janakakbase:narrative_1533320423326"
        }

        self.getImpl().predict_phenotype(self.getContext(), params)
        """
        """
        params = {
        
        "shock_id" : "5159651e-fdc1-4d2f-a7d3-d79948d2c9e7", #"8f55c4b4-750f-40c4-a973-f2c98825a18e",
        #"list_name" : "Genome_ID\ngenBankGO\nNC_003197\nShewanella_ondeisensis_MR-1_GenBank",#"",
        "classifier_name" : "whyNotDemo",
        "phenotypeclass" : "I'm predicting this",
        "workspace" : "sagoyal:narrative_1534292322496", #"janakakbase:narrative_1533153056355" #"janakakbase:narrative_1533153056355" "janakakbase:narrative_1533320423326"
        
        }
        """
        """
        params = {
        "shock_id": "bbf98000-860b-403a-bcd3-2fe1a10bd572",
        "list_name": "",
        "classifier_name": "demoCLF2_DecisionTreeClassifier_entropy",
        "phenotypeclass": "myDemoResp",
        "workspace" : "sagoyal:narrative_1534292322496"
        }
        self.getImpl().predict_phenotype(self.getContext(), params)
        """
    
    
    def test_upload_trainingset(self):
        """
        params = {
            "shock_id": "2b596bec-4327-4e4b-a094-c8a7b5af8b33",
            #"list_name": "Genome_ID Classification\nNC_003197    P\ngenBankGO   N\nShewanella_ondeisensis_MR-1_GenBank  N",
            #"list_name": "Genome_ID Classification\n204669.6    N\n234267.13    N\n240015.3 N\n1806.1   P\n83331.1  P\n106370.11    P\n164756.6 P\n216594.1 P\n227882.1 P\n233413.1 P\n246196.1 P\n262316.1 P\n266117.6 P\n272631.1 P\n313589.3 P\n321955.4 P\n350058.5 P\n351607.5 P\n369723.3 P\n391037.3 P\n443906.13    P\n446470.4 P\n471857.4 P\n479431.5 P\n526225.5 P\n698972.3 P\n994479.3 P\n1146883.3    P\n1206734.4    P\n1206739.4    P\n469378.5 P\n518634.7 P\n722911.3 P\n435830.3 P\n783.1    N\n955.1    N\n35793.1  N\n48935.1  N\n52598.3  N\n190650.1 N\n204722.1 N\n205920.8 N\n212042.5 N\n216596.1 N\n224911.1 N\n224914.1 N\n228405.5 N\n234826.3 N\n237727.3 N\n246200.3 N\n252305.3 N\n254945.3 N\n257363.1 N\n262698.3 N\n264203.3 N\n266779.1 N\n266834.1 N\n266835.1 N\n269484.4 N\n272944.1 N\n272947.1 N\n288000.5 N\n290633.1 N\n292414.1 N\n292805.3 N\n293614.3 N\n314225.3 N\n314232.3 N\n314254.3 N\n314260.3 N\n315456.3 N\n317655.9 N\n318586.4 N\n335992.3 N\n336407.4 N\n342108.5 N\n349163.4 N\n360095.3 N\n366602.3 N\n391165.8 N\n392499.4 N\n394221.5 N\n402881.6 N\n419610.8 N\n426117.3 N\n431944.4 N\n483179.3 N\n1112213.3    N\n398580.3 N\n224324.1 N\n648996.5 N\n868864.3 N\n123214.3 N\n66692.3  P\n93061.3  P\n176279.3 P\n220668.1 P\n224308.1 P\n235909.3 P\n257314.1 P\n272558.1 P\n272621.3 P\n279010.5 P\n279808.3 P\n281309.3 P\n299033.6 P\n315730.5 P\n321956.5 P\n321967.8 P\n342451.4 P\n387344.13    P\n405566.3 P\n420246.5 P\n169963.1 P\n272626.1 P\n386043.6 P\n262543.4 P\n1134785.3    P\n216432.3 N\n269798.12    P\n309807.19    P\n411154.5 N\n485917.5 N\n485918.5 N\n457391.3 N\n470145.6 N\n665954.3 N\n679190.3 N\n702438.4 N\n752555.5 N\n997879.3 N\n997891.3 N\n1042376.3    N\n376686.6 N\n402612.4 N\n486.1    N\n487.2    N\n36873.1  N\n76114.4  N\n204773.3 N\n216591.1 N\n228410.1 N\n232721.5 N\n243160.4 N\n243365.1 N\n264198.3 N\n265072.7 N\n266264.4 N\n267608.1 N\n272560.3 N\n292415.3 N\n296591.1 N\n335283.5 N\n339670.3 N\n350701.3 N\n391735.5 N\n395019.3 N\n395495.3 N\n398578.3 N\n999394.3 N\n115711.7 N\n218497.4 N\n227941.1 N\n243161.4 N\n264202.3 N\n759364.3 N\n194439.7 N\n517417.4 N\n316274.3 N\n324602.8 N\n383372.4 N\n292459.1 P\n1496.1   P\n138119.3 P\n195102.1 P\n203119.1 P\n212717.1 P\n246194.3 P\n264732.9 P\n290402.34    P\n293826.4 P\n335541.4 N\n349161.4 P\n357809.4 P\n411474.6 P\n413999.4 P\n431943.4 P\n485916.4 P\n522772.4 N\n68909.1  P\n243230.17    P?\n262724.1    N\n504728.4 N\n526227.4 N\n326298.3 N\n367737.4 N\n502025.5 N\n246197.19    N\n882.1    N\n891.1    N\n273121.1 N\n335543.6 N\n525897.4 N\n387093.4 N\n290397.13    N\n217.1    N\n85962.1  N\n192222.1 N\n235279.1 N\n306254.1 N\n306263.1 N\n306264.1 N\n360104.4 N\n360105.6 N\n360106.5 N\n360107.5 N\n382638.8 N\n525898.4 N\n706433.3 P\n469599.3 N\n519441.4 N\n354.1    N\n40324.1  N\n62977.3  N\n76869.3  N\n87626.3  N\n167879.3 N\n171440.1 N\n190485.1 N\n205918.4 N\n205922.3 N\n208963.3 N\n221988.1 N\n233412.1 N\n243159.3 N\n243233.4 N\n283942.3 N\n290398.4 N\n291331.3 N\n312309.3 N\n314275.3 N\n314276.3 N\n314282.3 N\n314283.3 N\n314288.3 N\n316275.9 N\n317025.3 N\n326442.4 N\n342610.3 N\n349521.5 N\n351348.5 N\n357804.5 N\n380703.5 N\n382245.6 N\n384676.6 N\n399739.6 N\n400667.4 N\n400668.6 N\n484022.4 N\n498211.3 N\n523791.4 N\n584.1    N\n615.1    N\n60480.16 N\n94122.5  N\n211586.1 N\n272620.3 N\n298386.1 N\n318161.14    N\n319224.13    N\n326297.7 N\n392500.3 N\n398579.3 N\n399599.3 N\n425104.3 N\n458817.3 N\n546273.3 N\n866778.4 N\n1048260.3    N\n63737.4  P\n103690.1 P\n993516.3 unknown\n189518.1   N\n12149.1  N\n273075.1 unknown\n178306.1   unknown\n272557.1   unknown\n399549.6   unknown\n243274.1   N\n240016.6 N",
            #"list_name": "Genome_ID Classification\n204669.6    N\n234267.13    N\n240015.3 N\n1806.1   P\n83331.1  P\n106370.11    P\n164756.6 P\n216594.1 P\n227882.1 P\n233413.1 P\n246196.1 P\n262316.1 P\n266117.6 P\n272631.1 P\n313589.3 P\n321955.4 P\n350058.5 P\n351607.5 P\n369723.3 P\n391037.3 P\n443906.13    P\n446470.4 P\n471857.4 P\n479431.5 P\n526225.5 P\n698972.3 P\n994479.3 P\n1146883.3    P\n1206734.4    P\n1206739.4    P\n469378.5 P\n518634.7 P\n722911.3 P\n435830.3 P\n783.1    N\n955.1    N\n35793.1  N\n48935.1  N\n52598.3  N\n190650.1 N\n204722.1 N\n205920.8 N\n212042.5 N\n216596.1 N\n224911.1 N\n224914.1 N\n228405.5 N\n234826.3 N\n237727.3 N\n246200.3 N\n252305.3 N\n254945.3 N\n257363.1 N\n262698.3 N\n264203.3 N\n266779.1 N\n266834.1 N\n266835.1 N\n269484.4 N\n272944.1 N\n272947.1 N\n288000.5 N\n290633.1 N\n292414.1 N\n292805.3 N\n293614.3 N\n314225.3 N\n314232.3 N\n314254.3 N\n314260.3 N\n315456.3 N\n317655.9 N\n318586.4 N\n335992.3 N\n336407.4 N\n342108.5 N\n349163.4 N\n360095.3 N\n366602.3 N\n391165.8 N\n392499.4 N\n394221.5 N\n402881.6 N\n419610.8 N\n426117.3 N\n431944.4 N\n483179.3 N\n1112213.3    N\n398580.3 N\n224324.1 N\n648996.5 N\n868864.3 N\n123214.3 N\n66692.3  P\n93061.3  P\n176279.3 P\n220668.1 P\n224308.1 P\n235909.3 P\n257314.1 P\n272558.1 P\n272621.3 P\n279010.5 P\n279808.3 P\n281309.3 P\n299033.6 P\n315730.5 P\n321956.5 P\n321967.8 P\n342451.4 P\n387344.13    P\n405566.3 P\n420246.5 P\n169963.1 P\n272626.1 P\n386043.6 P\n262543.4 P\n1134785.3    P\n216432.3 N\n269798.12    P\n309807.19    P\n411154.5 N\n485917.5 N\n485918.5 N\n457391.3 N\n470145.6 N\n665954.3 N\n679190.3 N\n702438.4 N\n752555.5 N\n997879.3 N\n997891.3 N\n1042376.3    N\n376686.6 N\n402612.4 N\n486.1    N\n487.2    N\n36873.1  N\n76114.4  N\n204773.3 N\n216591.1 N\n228410.1 N\n232721.5 N\n243160.4 N\n243365.1 N\n264198.3 N\n265072.7 N\n266264.4 N\n267608.1 N\n272560.3 N\n292415.3 N\n296591.1 N\n335283.5 N\n339670.3 N\n350701.3 N\n391735.5 N\n395019.3 N\n395495.3 N\n398578.3 N\n999394.3 N\n115711.7 N\n218497.4 N\n227941.1 N\n243161.4 N\n264202.3 N\n759364.3 N\n194439.7 N\n517417.4 N\n316274.3 N\n324602.8 N\n383372.4 N\n292459.1 P\n1496.1   P\n138119.3 P\n195102.1 P\n203119.1 P\n212717.1 P\n246194.3 P\n264732.9 P\n290402.34    P\n293826.4 P\n335541.4 N\n349161.4 P\n357809.4 P\n411474.6 P\n413999.4 P\n431943.4 P\n485916.4 P\n522772.4 N\n68909.1  P\n243230.17    P\n262724.1 N\n504728.4 N\n526227.4 N\n326298.3 N\n367737.4 N\n502025.5 N\n246197.19    N\n882.1    N\n891.1    N\n273121.1 N\n335543.6 N\n525897.4 N\n387093.4 N\n290397.13    N\n217.1    N\n85962.1  N\n192222.1 N\n235279.1 N\n306254.1 N\n306263.1 N\n306264.1 N\n360104.4 N\n360105.6 N\n360106.5 N\n360107.5 N\n382638.8 N\n525898.4 N\n706433.3 P\n469599.3 N\n519441.4 N\n354.1    N\n40324.1  N\n62977.3  N\n76869.3  N\n87626.3  N\n167879.3 N\n171440.1 N\n190485.1 N\n205918.4 N\n205922.3 N\n208963.3 N\n221988.1 N\n233412.1 N\n243159.3 N\n243233.4 N\n283942.3 N\n290398.4 N\n291331.3 N\n312309.3 N\n314275.3 N\n314276.3 N\n314282.3 N\n314283.3 N\n314288.3 N\n316275.9 N\n317025.3 N\n326442.4 N\n342610.3 N\n349521.5 N\n351348.5 N\n357804.5 N\n380703.5 N\n382245.6 N\n384676.6 N\n399739.6 N\n400667.4 N\n400668.6 N\n484022.4 N\n498211.3 N\n523791.4 N\n584.1    N\n615.1    N\n60480.16 N\n94122.5  N\n211586.1 N\n272620.3 N\n298386.1 N\n318161.14    N\n319224.13    N\n326297.7 N\n392500.3 N\n398579.3 N\n399599.3 N\n425104.3 N\n458817.3 N\n546273.3 N\n866778.4 N\n1048260.3    N\n63737.4  P\n103690.1 P\n189518.1 N\n12149.1  N\n243274.1 N\n240016.6 N",
            "list_name": "Genome_ID Classification\nAcetivibrio_ethanolgignens  N\nAggregatibacter_actinomycetemcomitans_serotype_b_str._SCC4092    P\nAfipia_felis_ATCC_53690  N",
            "phenotypeclass": "DoesnotMatter",
            "training_set_out": "Tset3",
            "workspace" : "avi2:narrative_1534266317055"#"janakakbase:narrative_1534966345663"#"sagoyal:narrative_1533659119242" #sagoyal:narrative_1534292322496"
        }

        self.getImpl().upload_trainingset(self.getContext(), params)
        """

    def test_2build_classifier(self):
        """
        params = {
            "trainingset_name": "my290",
            "phenotypeclass": "myPheno",
            "classifier": 'KNeighborsClassifier',#"DecisionTreeClassifier",#'KNeighborsClassifier',#'run_all', #"KNeighborsClassifier",
            "attribute": "functional_roles",
            "save_ts": 1,
            "classifier_out": "forMRole",
            "workspace" : "janakakbase:narrative_1534966345663"#"sagoyal:narrative_1533659119242"#"sagoyal:narrative_1534292322496"#"sagoyal:narrative_1533659119242"
        }
        """

        """
        params = {
        "trainingset_name": "myTSet",
        "phenotypeclass": "Gram",
        "classifier": "DecisionTreeClassifier",
        "attribute": "functional_roles",
        "save_ts": 1,
        "classifier_out": "myDTCLF",
        "workspace" : "avi2:narrative_1534266317055"
        }
        """

        """
        params = {
        "trainingset_name": "Tset3",
        "save_ts": 1,
        "classifier_out": "DTCFL",
        "attribute": "functional_roles",
        "phenotypeclass": "Gram",
        "classifier": "SVM",
        "workspace" : "avi2:narrative_1534266317055"
        }

        self.getImpl().build_classifier(self.getContext(), params)
        """

    """
    def test_2predict_phenotype(self):
        params = {
        "shock_id": "bbf98000-860b-403a-bcd3-2fe1a10bd572",
        "list_name": "Genome_ID Classification\nShewanella_ondeisensis_MR-1_GenBank Aerobic\ngenBankGO  Anaerobic\nNC_003197    Facultative\nGCF_000010525.1    Facultative\nGCF_000007365.1    Aerobic\nGCF_000007725.1    Anaerobic\nGCF_000009605.1  Aerobic\nGCF_000021065.1    Anaerobic\nGCF_000021085.1  Facultative\nGCF_000090965.1    Facultative\nGCF_000174075.1    Aerobic\nGCF_000183225.1    Aerobic\nGCF_000183245.1    Facultative\nGCF_000183285.1    Facultative\nGCF_000183305.1    Anaerobic\nGCF_000217635.1  Aerobic\nGCF_000225445.1    Aerobic\nGCF_000225465.1    Anaerobic\nGCF_000521525.1  Anaerobic\nGCF_000521545.1  Aerobic\nGCF_000521565.1    Aerobic\nGCF_000521585.1    Facultative\nGCF_001280225.1    Anaerobic\nGCF_001648115.1  Facultative",
        "classifier_name": "Big25",
        "phenotypeclass": "myPhenotype",
        "workspace" : "sagoyal:narrative_1534292322496"
        }

        self.getImpl().predict_phenotype(self.getContext(), params)
    """

    def test_3build_classifier(self):
        """
        params =     {
            "description": "I can make this whatever I want it to be",
            "save_ts": 1,
            "trainingset_name": "BIG25",
            "phenotypeclass": "Respiration",
            "classifier": "run_all",
            "k_nearest_neighbors": {
                "n_neighbors": 5,
                "weights": "uniform",
                "algorithm": "auto",
                "leaf_size": 30,
                "p": 2,
                "metric": "minkowski",
                "metric_params": "",
                "knn_n_jobs": 1
            },
            "gaussian_nb": {
                "priors": ""
            },
            "logistic_regression": {
                "penalty": "l2",
                "dual": "False",
                "lr_tolerance": 0.0001,
                "lr_C": 1,
                "fit_intercept": "True",
                "intercept_scaling": 1,
                "lr_class_weight": "",
                "lr_random_state": 0,
                "lr_solver": "newton-cg",
                "lr_max_iter": 100,
                "multi_class": "ovr",
                "lr_verbose": 0,
                "lr_warm_start": "False",
                "lr_n_jobs": 1
            },
            "decision_tree_classifier": {
                "criterion": "gini",
                "splitter": "best",
                "max_depth": None,
                "min_samples_split": 2,
                "min_samples_leaf": 1,
                "min_weight_fraction_leaf": 0,
                "max_features": "",
                "dt_random_state": 0,
                "max_leaf_nodes": None,
                "min_impurity_decrease": 0,
                "dt_class_weight": "",
                "presort": "False"
            },
            "support_vector_machine": {
                "svm_C": 1,
                "kernel": "linear",
                "degree": 3,
                "gamma": "auto",
                "coef0": 0,
                "probability": "False",
                "shrinking": "True",
                "svm_tolerance": 0.001,
                "cache_size": 200,
                "svm_class_weight": "",
                "svm_verbose": "False",
                "svm_max_iter": -1,
                "decision_function_shape": "ovr",
                "svm_random_state": 0
            },
            "neural_network": {
                "hidden_layer_sizes": "(100,)",
                "activation": "relu",
                "mlp_solver": "adam",
                "alpha": 0.0001,
                "batch_size": "auto",
                "learning_rate": "constant",
                "learning_rate_init": 0.001,
                "power_t": 0.05,
                "mlp_max_iter": 200,
                "shuffle": "True",
                "mlp_random_state": 0,
                "mlp_tolerance": 0.0001,
                "mlp_verbose": "False",
                "mlp_warm_start": "False",
                "momentum": 0.9,
                "nesterovs_momentum": "True",
                "early_stopping": "False",
                "validation_fraction": 0.1,
                "beta_1": 0.9,
                "beta_2": 0.999,
                "epsilon": 1e-8
            },
            "attribute": "functional_roles",
            "classifier_out": "myCLF",
            "workspace" : "sagoyal:narrative_1534292322496"
        }
        """
        """
        params = {
        "description": "I can make this whatever I want it to be",
        "save_ts": 1,
        "trainingset_name": "BIG25",
        "phenotypeclass": "Respiration",
        "classifier": "run_all",
        "k_nearest_neighbors": None,
        "gaussian_nb": None,
        "logistic_regression": None,
        "decision_tree_classifier": None,
        "support_vector_machine": None,
        "neural_network": None,
        "attribute": "functional_roles",
        "classifier_out": "myCLF",
        "workspace" : "sagoyal:narrative_1534292322496"
        }
        """

        """
        params = {
        "neural_network": None,
        "save_ts": 1,
        "description": "This my new Description",
        "attribute": "functional_roles",
        "phenotypeclass": "Respiration",
        "trainingset_name": "myNewTset",
        "logistic_regression": None,
        "gaussian_nb": None,
        "decision_tree_classifier": None,
        "classifier_out": "allCLF",
        "support_vector_machine": None,
        "classifier": "run_all",
        "k_nearest_neighbors": None,
        "workspace" : "sagoyal:narrative_1534292322496"
        }
        """

        params = {
        "save_ts": 1,
        "description": "my Phylum Classifier",
        "trainingset_name": "myPhylumTset",
        "phenotypeclass": "Phylum",
        "classifier": "run_all",
        "attribute": "functional_roles",
        "k_nearest_neighbors": None,
        "gaussian_nb": None,
        "logistic_regression": None,
        "decision_tree_classifier": None,
        "support_vector_machine": None,
        "neural_network": None,
        "classifier_out": "myPhylumCLF",
        "workspace" : "sagoyal:narrative_1534259992668"
        }

        self.getImpl().build_classifier(self.getContext(), params)
