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
    

        params = {
            "shock_id": "2b596bec-4327-4e4b-a094-c8a7b5af8b33",
            "list_name": "Genome_ID Classification\nNC_003197    P\ngenBankGO   N\nShewanella_ondeisensis_MR-1_GenBank  N",
            "phenotypeclass": "DoesnotMatter",
            "training_set_out": "Tset3",
            "workspace" : "sagoyal:narrative_1533659119242" #sagoyal:narrative_1534292322496"
        }

        self.getImpl().upload_trainingset(self.getContext(), params)
    

    """
    def test_2build_classifier(self):
        params = {
            "trainingset_name": "Tset3",
            "phenotypeclass": "myPheno",
            "classifier": "DecisionTreeClassifier",
            "attribute": "functional_roles",
            "save_ts": 0,
            "classifier_out": "newClf3T",
            "workspace" : "sagoyal:narrative_1533659119242"#"sagoyal:narrative_1534292322496"#"sagoyal:narrative_1533659119242"
        }

        self.getImpl().build_classifier(self.getContext(), params)
    
    
    def test_2predict_phenotype(self):
        params = {
        "shock_id": "bbf98000-860b-403a-bcd3-2fe1a10bd572",
        "list_name": "Genome_ID\ngenBankGO\nNC_003197\nShewanella_ondeisensis_MR-1_GenBank",
        "classifier_name": "newClf3T",
        "phenotypeclass": "myPhenotypeforTrain",
        "workspace" : "sagoyal:narrative_1533659119242"
        }

        self.getImpl().predict_phenotype(self.getContext(), params)
    """

