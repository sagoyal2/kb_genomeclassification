from __future__ import division


from KBaseReport.KBaseReportClient import KBaseReport
from DataFileUtil.DataFileUtilClient import DataFileUtil
from RAST_SDK.RAST_SDKClient import RAST_SDK
from biokbase.workspace.client import Workspace as workspaceService


class kb_genomeclfUtils(object):
	def __init__(self, config):

		self.workspaceURL = config['workspaceURL']
		self.scratch = config['scratch']
		self.callback_url = config['callback_url']

		self.ctx = config['ctx']

		self.dfu = DataFileUtil(self.callback_url)
		self.rast = RAST_SDK(self.callback_url)
		self.ws_client = workspaceService(self.workspaceURL)


	#### MAIN Methods below are called from KBASE apps ###
	
	#return htmloutput_name, classifier_info_list, weight_list
	def fullClassify(self, params, current_ws):

		print(params)
		print(current_ws)

	#return htmloutput_name, predictions_mapping
	def fullPredict(self, params, current_ws):
		pass

	#return htmloutput_name, classifier_training_set_mapping
	def fullUpload(self, params, current_ws):
		pass
