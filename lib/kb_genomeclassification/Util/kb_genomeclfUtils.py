# -*- coding: utf-8 -*-
#BEGIN_HEADER
# The header block is where all import statments should live

from __future__ import division

import os
import re
import sys
import ast
import uuid
import xlrd
import json
import random
import codecs
import graphviz
import StringIO
import xlsxwriter

import pickle
import numpy as np
import pandas as pd
import seaborn as sns

from io import open

import itertools
from itertools import izip

#classifier models
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier

#additional classifier methods
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold

#Switch the backend to plot confusion matrix figure
import matplotlib.pyplot as plt
plt.switch_backend('agg')
#needed to display ipython notebook
#%matplotlib inlines

from KBaseReport.KBaseReportClient import KBaseReport
from DataFileUtil.DataFileUtilClient import DataFileUtil
from RAST_SDK.RAST_SDKClient import RAST_SDK
from biokbase.workspace.client import Workspace as workspaceService



class kb_genomeclfUtils(object):
	"""docstring for ClassName"""
	def __init__(self, config):

		self.workspaceURL = config['workspaceURL']
		self.scratch = config['scratch']
		self.callback_url = config['callback_url']

		self.ctx = config['ctx']

		self.dfu = DataFileUtil(self.callback_url)
		self.rast = RAST_SDK(self.callback_url)
		self.ws_client = workspaceService(self.workspaceURL)

		self.list_name = []
		self.list_statistics = []


	#### MAIN Methods below are called from KBASE apps ###
	def fullClassify(self, params, current_ws):
		"""
		args:
		---params from build_classifier
		---current_ws which is a narrative enviornment variable necessary to access the KBase workspace
		does:
		---first it sets up the Xs and Ys in dataframe (the columns are masterRole which are all the function rolls and the indexes are the Genome_IDs the rest is a matrix of 1's and 0s
														that that have a classification stored in the last column)
		---runs the classifers based on the classifier_type (either run_all, Decision Tree, or single selection)
			---if run_all or Decision Tree is selected then a second 'tab' is also created in the html page base on tuning of the Decision Tree
		---creates html pages with 'tabs' that show the main classifcation and statistics on the classification
		return:
		---an htmloutput_name which is the name of the html report that get created from makeHtmlReport
		--- 
		"""
		
		#SETUP of X & Y trainging and testing sets

		#params = self.editBuildArguments(params)

		print ('Frist Print')
		#print(self.editBuildArguments(params))

		params = self.editBuildArguments(params)

		print("here are my curret params:")
		print(params)

		#print "Below is the classifierAdvanced_params"
		#print(self.editBuildArguments(params)["classifierAdvanced_params"])

		toEdit_all_classifications, training_set_ref, num_classes, listOfRefs = self.unloadGenomeClassifierTrainingSet(current_ws, params['trainingset_name'])
		listOfNames, all_classifications = self.intake_method(toEdit_all_classifications)
		all_attributes, master_Role = self.get_wholeClassification(listOfNames, current_ws, params['attribute'], refs = listOfRefs)
	

		if params.get('save_ts') != 1:
			training_set_ref = 'User Denied'


		#Load in 'cached' data from the data folder
		
		"""
		training_set_ref = '35424/384/1'
		#pickle_in = open("/kb/module/data/Classifications_DF.pickle", "rb")
		pickle_in = open("/kb/module/data/myPhylumDF.pickle", "rb")
		all_classifications = pickle.load(pickle_in)
		listOfNames = all_classifications.index

		pickle_in = open("/kb/module/data/fromKBASE_Phylum_MR.pickle", "rb")
		master_Role = pickle.load(pickle_in)

		pickle_in = open("/kb/module/data/fromKBASE_Phylum_attributes.pickle", "rb")
		all_attributes = pickle.load(pickle_in)
		"""

		all_attributes = all_attributes.T[listOfNames].T
		

		#mapping of string classes to integers
		correctClassifications_list = []

		for index in range(len(all_classifications['Classification'])):
			correctClassifications_list.append(all_classifications['Classification'][index])
			
		class_list = list(set(correctClassifications_list))

		my_mapping = {} #my_mapping = {'aerobic': '0', 'anaerobic': '1', 'facultative': '2'}
		for current_class,num in zip(class_list, range(0, len(class_list))):
			my_mapping[current_class] = num

		my_class_mapping = []

		for index in range(len(correctClassifications_list)):
			my_class_mapping.append(my_mapping[correctClassifications_list[index]])
			
		print(len(my_class_mapping))

		all_attributes = all_attributes.values.astype(int)
		all_classifications = np.array(my_class_mapping)
		
		"""
		training_set_ref = '35424/320/1'
		all_attributes = np.load("/kb/module/data/random_attribute_array.npy")
		all_classifications = np.load("/kb/module/data/random_classification_array.npy")
		class_list = ["A", "B", "C", "D", "E", "F", "G"]
		my_mapping = {}

		pickle_in = open("/kb/module/data/fromKBASE_MR.pickle", "rb")
		master_Role = pickle.load(pickle_in)
		"""

		"""
		full_dataFrame = pd.concat([all_attributes, all_classifications], axis = 1, sort=True)

		print "Below is full_dataFrame"
		print full_dataFrame

		class_list = list(set(full_dataFrame['Classification']))

		#create a mapping
		my_mapping = {}
		for current_class,num in zip(class_list, range(0, len(class_list))):
			my_mapping[current_class] = num

		for index in full_dataFrame.index:
			full_dataFrame.at[index, 'Classification'] = my_mapping[full_dataFrame.at[index, 'Classification']]

		all_classifications = full_dataFrame['Classification']

		print "Below is full_attribute_array and full_classification_array"

		all_attributes = all_attributes.values.astype(int)
		all_classifications = all_classifications.values.astype(int)

		print "Below is full_attribute_array and full_classification_array round 2"
		print all_attributes
		print all_classifications
		"""

		#np.save(os.path.join(self.scratch,'full_attribute_array.npy'), all_attributes)
		#np.save(os.path.join(self.scratch, 'full_classification_array.npy'), all_classifications)

		#pickle_out = open(os.path.join(self.scratch,"attribute_list.pickle"), "wb")
		#pickle.dump(master_Role, pickle_out)

		self.createHoldingDirs()

		ctx = self.ctx
		token = ctx['token']

		classifier_type = params.get('classifier')
		global_target = params.get('phenotypeclass')
		classifier_name = params.get('classifier_out')

		folderhtml1 = "html1folder/"
		folderhtml2 = "html2folder/"
		folderhtml4 = "html4folder/"


		train_index = []
		test_index = []

		splits = 2 #10 #10 #2

		#This cross-validation object is a variation of KFold that returns stratified folds. The folds are made by preserving the percentage of samples for each class.
		skf = StratifiedKFold(n_splits=splits, random_state=0, shuffle=True)
		for train_idx, test_idx in skf.split(all_attributes, all_classifications):
			train_index.append(train_idx)
			test_index.append(test_idx)

		classifierTest_params = {
				'ctx' : ctx,
				'current_ws' : current_ws,
				#'classifier' : ,
				'classifier_type' : classifier_type,
				'classifier_name' : classifier_name,
				'my_mapping' : my_mapping,
				'master_Role' : master_Role,
				'splits' : splits,
				'train_index' : train_index,
				'test_index' : test_index,
				'all_attributes' : all_attributes,
				'all_classifications' : all_classifications,
				'class_list' : class_list,
				'htmlfolder' : folderhtml1,
				'print_cfm' : True,
				'training_set_ref' : training_set_ref,
				'description' : params["description"]
				}

		#RUNNING the classifiers depending on classifier_type

		if classifier_type == u"run_all":

			listRunAll = ['KNeighborsClassifier', 'GaussianNB', 'LogisticRegression', 'DecisionTreeClassifier', 'SVM', 'NeuralNetwork']
			all_advanced = ["k_nearest_neighbors", "gaussian_nb", "logistic_regression", "decision_tree_classifier", "support_vector_machine", "neural_network"]

			for run, advanced in izip(listRunAll, all_advanced):
				#classifierTest_params['classifier'] = self.whichClassifier(run)
				classifierTest_params['classifier'] = self.whichClassifierAdvanced(run, params[advanced])
				classifierTest_params['classifier_type'] = run
				classifierTest_params['classifier_name'] = classifier_name + u'_' + run
				classifierTest_params['htmlfolder'] = folderhtml1

				self.classifierTest(classifierTest_params)

			best_classifier_str = self.to_HTML_Statistics(class_list, classifier_name)

			print("This is the best_classifier_str")
			print(best_classifier_str)

			#best_classifier_str = classifier_name+u"_LogisticRegression"
			best_classifier_type = best_classifier_str[classifier_name.__len__() + 1:] #extract just the classifier_type aka. "LogisticRegression" from "myName_LogisticRegression"
			best_classifier_type_index = listRunAll.index(best_classifier_type)
			#to create another "best in html2"
			
			#classifierTest_params['classifier'] = self.whichClassifier(best_classifier_type)
			classifierTest_params['classifier'] = self.whichClassifierAdvanced(best_classifier_type, params[all_advanced[best_classifier_type_index]], True)
			classifierTest_params['classifier_type'] = best_classifier_type
			classifierTest_params['classifier_name'] = classifier_name+u"_" + best_classifier_type
			classifierTest_params['htmlfolder'] = folderhtml2
			self.classifierTest(classifierTest_params)

			self.html_report_1(global_target, classifier_type, classifier_name, params['phenotypeclass'], num_classes, best_classifier_str= best_classifier_str)

			classifierTest_params['classifier_name'] = classifier_name
			classifierTest_params['htmlfolder'] = folderhtml2
			self.tune_Decision_Tree(classifierTest_params, best_classifier_str)
			self.html_report_2(global_target, classifier_name, num_classes, best_classifier_str)

			classifierTest_params['classifier'], estimators_inHTML = self.ensembleCreation(params["ensemble_model"], params)
			
			if classifierTest_params['classifier'] == "No_Third":
				htmloutput_name = self.html_dual_12()

			else:
				classifierTest_params['classifier_type'] = "Ensemble_Model"
				classifierTest_params['classifier_name'] = classifier_name+u"_" + "Ensemble_Model"
				classifierTest_params['htmlfolder'] = folderhtml4
				self.classifierTest(classifierTest_params)

				self.to_HTML_Statistics(class_list, classifier_name, known = classifierTest_params['classifier_name'], for_ensemble = True)

				self.html_report_4(global_target, classifier_name, estimators_inHTML, num_classes)

				htmloutput_name = self.html_dual_123()

		elif classifier_type == u"DecisionTreeClassifier":

			#classifierTest_params['classifier'] = self.whichClassifier(classifier_type)
			classifierTest_params['classifier'] = self.whichClassifierAdvanced(classifier_type, params["classifierAdvanced_params"])
			self.classifierTest(classifierTest_params)

			self.to_HTML_Statistics(class_list, classifier_name)
			
			self.html_report_1(global_target, classifier_type, classifier_name, params['phenotypeclass'], num_classes)


			classifierTest_params['htmlfolder'] = folderhtml2
			self.tune_Decision_Tree(classifierTest_params)
			
			self.html_report_2(global_target, classifier_name, num_classes)
			htmloutput_name = self.html_dual_12()

		else:
			#classifierTest_params['classifier'] = self.whichClassifier(classifier_type)
			classifierTest_params['classifier'] = self.whichClassifierAdvanced(classifier_type, params["classifierAdvanced_params"])
			self.classifierTest(classifierTest_params)

			self.to_HTML_Statistics(class_list, classifier_name)
			self.html_report_1(global_target, classifier_type, classifier_name, params['phenotypeclass'], num_classes)
			htmloutput_name = self.html_nodual("forHTML")

		return htmloutput_name
		
	def fullPredict(self, params, current_ws):
		"""
		args:
		---params from predict_phenotype
		---current_ws which is a narrative enviornment variable necessary to access the KBase workspace
		does:
		---first it sets up the Xs in dataframe (the columns are masterRole which are all the function rolls and the indexes are the Genome_IDs)
			--- The Xs in this case is the testing data or unclassified data
		---based on the user selected classifier makes predictions on the data with a column for the probability of the prediction being correct
		return:
		---an htmloutput_name which is the name of the html report that get created from makeHtmlReport
		--- 
		"""

		ctx = self.ctx
		token = ctx['token']

		self.createHoldingDirs(False)

		current_ws = params.get('workspace')
		classifier_name = params.get('classifier_name')
		target = params.get('phenotypeclass')

		#getting user selected classifer from workspace and then "unpickling & unbase64ing" it into a usable classifier
		classifier_object = self.ws_client.get_objects([{'workspace':current_ws, 'name':classifier_name}])

		#base64str = str(classifier_object[0]['data']['classifier_data'])
		clf_shock_id = classifier_object[0]['data']['classifier_handle_ref']
		clf_shock_id  = str(current_ws + '/' + classifier_name + ';' + clf_shock_id)
		clf_shock_id = clf_shock_id.split(':')[-1]
		print(clf_shock_id)
		clf_file_path = self._download_shock(handle_id=clf_shock_id)

		master_Role = classifier_object[0]['data']['attribute_data']
		my_mapping = classifier_object[0]['data']['class_list_mapping']

		#after_classifier = pickle.loads(codecs.decode(base64str.encode(), "base64"))
		pickle_in = open(clf_file_path, "rb")
		after_classifier = pickle.load(pickle_in)

		if params.get('list_name'):
			#checks if empty string bool("") --> False
			toEdit_all_classifications = self.incaseList_Names(params.get('list_name'))
			#listOfNames = self.intake_method(toEdit_all_classifications, for_predict = True)

			print("this is toEdit_all_classifications")
			print(toEdit_all_classifications)
			
			(missingGenomes, inKBASE) = self.createGenomeClassifierTrainingSet(current_ws, params['Annotated'], just_DF = toEdit_all_classifications, for_predict = True)
			#Will error out if inKBASE == 0
			all_attributes = self.get_wholeClassification(inKBASE, current_ws, params['attribute'], master_Role = master_Role ,for_predict = True)
		else:
			file_path = self._download_shock(shock_id = params.get('shock_id'))
			#listOfNames, all_classifications = self.intake_method(just_DF = pd.read_excel(file_path))
			(missingGenomes, inKBASE)  = self.createGenomeClassifierTrainingSet(current_ws,params['Annotated'], just_DF = pd.read_excel(file_path, dtype=str), for_predict = True)
			#Will error out if inKBASE == 0
			all_attributes = self.get_wholeClassification(inKBASE, current_ws, params['attribute'], master_Role = master_Role ,for_predict = True)

		# if params.get('list_name'):
		# 	#checks if empty string bool("") --> False
		# 	print ("taking this path rn")
		# 	print(params)
		# 	toEdit_all_classifications = self.incaseList_Names(params.get('list_name'))
		# 	(missingGenomes, inKBASE, inKBASE_Classification) = self.createGenomeClassifierTrainingSet(current_ws,params['Annotated'], just_DF = toEdit_all_classifications, for_predict = True)
		# 	#self.newReferencetoGenome(current_ws, params['description'], params['training_set_out'], inKBASE, inKBASE_Classification)
		# 	#self.workRAST(current_ws, just_DF = toEdit_all_classifications)
		# 	#listOfNames, all_classifications = self.intake_method(toEdit_all_classifications)
		# 	#all_attributes, master_Role = self.get_wholeClassification(listOfNames, current_ws)
		# else:
		# 	print("printing the params to see RAST")
		# 	print(params)
		# 	#file_path = self._download_shock(params.get('shock_id'))
		# 	(missingGenomes, inKBASE, inKBASE_Classification) = self.createGenomeClassifierTrainingSet(current_ws,params['Annotated'], just_DF = pd.read_excel(file_path), for_predict = True)
		# 	#self.newReferencetoGenome(current_ws, params['description'], params['training_set_out'], inKBASE, inKBASE_Classification)
		# 	#self.workRAST(current_ws, just_DF = pd.read_excel(file_path))
		# 	#listOfNames, all_classifications = self.intake_method(just_DF = pd.read_excel(file_path))
		# 	#all_attributes, master_Role = self.get_wholeClassification(listOfNames, current_ws)

		#PREDICTIONS on new data that needs to be classified
		after_classifier_result = after_classifier.predict(all_attributes)

		after_classifier_result_forDF = []

		for current_result in after_classifier_result:
			after_classifier_result_forDF.append(my_mapping.keys()[my_mapping.values().index(current_result)])


		#after_classifier_df = pd.DataFrame(after_classifier_result_forDF, index=all_attributes.index, columns=[target])
		allProbs = after_classifier.predict_proba(all_attributes)
		maxEZ = np.amax(allProbs, axis=1)

		# after_classifier_df = pd.DataFrame.from_dict({'Genome Id': all_attributes.index,target: after_classifier_result_forDF})
		# after_classifier_df = after_classifier_df[['Genome Id', target]]

		#create a column for the probability of a prediction being accurate
		# allProbs = after_classifier.predict_proba(all_attributes)
		# maxEZ = np.amax(allProbs, axis=1)
		# maxEZ_df = pd.DataFrame(maxEZ, index=all_attributes.index, columns=["Probability"])

		predict_table_pd = pd.DataFrame.from_dict({'Genome Id': all_attributes.index, target: after_classifier_result_forDF, "Probability": maxEZ})
		#predict_table_pd = predict_table_pd.set_index('Genome Id')
		predict_table_pd = predict_table_pd[['Genome Id', target, "Probability"]]



		predict_table_pd.to_html(os.path.join(self.scratch, 'forSecHTML', 'html3folder', 'results.html'), index=False, table_id="results", classes =["table", "table-striped", "table-bordered"])

		#you can also save down table as text file or csv
		"""
		#txt
		np.savetxt(r'/kb/module/work/tmp/np.txt', predict_table_pd.values, fmt='%d')

		#csv
		predict_table_pd.to_csv(r'/kb/module/work/tmp/pandas.txt', header=None, index=None, sep=' ', mode='a')
		"""
		self.html_report_3(missingGenomes, params['phenotypeclass'])
		htmloutput_name = self.html_nodual("forSecHTML")

		return htmloutput_name

	def fullUpload(self, params, current_ws):

		out_path = os.path.join(self.scratch, 'forZeroHTML')
		os.makedirs(out_path)

		if params.get('list_name'):
			#checks if empty string bool("") --> False
			print ("taking this path rn")
			print(params)
			toEdit_all_classifications = self.incaseList_Names(params.get('list_name'))
			(missingGenomes, inKBASE, inKBASE_Classification) = self.createGenomeClassifierTrainingSet(current_ws,params['Annotated'], just_DF = toEdit_all_classifications)
			self.newReferencetoGenome(current_ws, params['description'], params['training_set_out'], inKBASE, inKBASE_Classification)
			#self.workRAST(current_ws, just_DF = toEdit_all_classifications)
			#listOfNames, all_classifications = self.intake_method(toEdit_all_classifications)
			#all_attributes, master_Role = self.get_wholeClassification(listOfNames, current_ws)
		else:
			print("printing the params to see RAST")
			print(params)
			#file_path = self._download_shock(params.get('shock_id'))
			(missingGenomes, inKBASE, inKBASE_Classification) = self.createGenomeClassifierTrainingSet(current_ws,params['Annotated'], just_DF = pd.read_excel(os.path.join(os.path.sep,"staging",file_path)))
			self.newReferencetoGenome(current_ws, params['description'], params['training_set_out'], inKBASE, inKBASE_Classification)
			#self.workRAST(current_ws, just_DF = pd.read_excel(file_path))
			#listOfNames, all_classifications = self.intake_method(just_DF = pd.read_excel(file_path))
			#all_attributes, master_Role = self.get_wholeClassification(listOfNames, current_ws)

		self.html_report_0(missingGenomes, params['phenotypeclass'])
		htmloutput_name = self.html_nodual("forZeroHTML")

		return htmloutput_name

	def workRAST(self, current_ws, just_DF):

		listintGNames = just_DF['Genome_ID']
		
		#vigorous string matching izip(self.list_name, self.list_statistics)
		listGNames = list(map(str, listintGNames))
		for string, index in izip(listGNames, range(len(listGNames))):
			listGNames[index] = string.replace(" ", "")

		print("we are printing just_DF")
		print(listGNames)

		for index in range(len(listGNames)):

			params_RAST =	{
			"workspace": current_ws,#"sagoyal:narrative_1536939130038",
			"input_genome": listGNames[index],
			"output_genome": listGNames[index]+".RAST",
			"call_features_rRNA_SEED": 0,
			"call_features_tRNA_trnascan": 0,
			"call_selenoproteins": 0,
			"call_pyrrolysoproteins": 0,
			"call_features_repeat_region_SEED": 0,
			"call_features_strep_suis_repeat": 0,
			"call_features_strep_pneumo_repeat": 0,
			"call_features_crispr": 0,
			"call_features_CDS_glimmer3": 0,
			"call_features_CDS_prodigal": 0,
			"annotate_proteins_kmer_v2": 1,
			"kmer_v1_parameters": 1,
			"annotate_proteins_similarity": 1,
			"retain_old_anno_for_hypotheticals": 0,
			"resolve_overlapping_features": 0,
			"call_features_prophage_phispy": 0
			}

			output = self.rast.annotate_genomes(params_RAST)

		print(output)

		# params_RAST =  {
		# "workspace": "sagoyal:narrative_1536939130038",#"sagoyal:narrative_1534292322496",
		# "input_genomes": ["36230/305/3", "36230/304/3"], #[]
		# "genome_text": "",#my_genome_text,
		# "call_features_rRNA_SEED": 0,
		# "call_features_tRNA_trnascan": 0,
		# "call_selenoproteins": 0,
		# "call_pyrrolysoproteins": 0,
		# "call_features_repeat_region_SEED": 0,
		# "call_features_insertion_sequences": 0,
		# "call_features_strep_suis_repeat": 0,
		# "call_features_strep_pneumo_repeat": 0,
		# "call_features_crispr": 0,
		# "call_features_CDS_glimmer3": 0,
		# "call_features_CDS_prodigal": 0,
		# "annotate_proteins_kmer_v2": 1,
		# "kmer_v1_parameters": 1,
		# "annotate_proteins_similarity": 1,
		# "retain_old_anno_for_hypotheticals": 0,
		# "resolve_overlapping_features": 0,
		# "call_features_prophage_phispy": 0
		# }


	def makeHtmlReport(self, htmloutput_name, current_ws, which_report, description, for_predict = False):
		"""
		args:
		---htmloutput_name the name of the html file 'something.html'
		---which_report whether from clf_Runner or pred_Runner
		does:
		---using shock, zips and uploads an html folder to kbase app that is then viewable through the reports window
		return:
		---report_output the html report itself as a ws object that is viewable after the Kbase app runs
		--- 
		"""

		ctx = self.ctx
		token = ctx['token']
		uuid_string = str(uuid.uuid4())

		if which_report == 'clf_Runner':
			saved_html = 'forHTML'
		
		if which_report == 'pred_Runner':
			saved_html = 'forSecHTML'

		if which_report == 'upload_Runner':
			saved_html = 'forZeroHTML'

		output_directory = os.path.join(self.scratch, saved_html)
		report_shock_id = self.dfu.file_to_shock({'file_path': output_directory,'pack': 'zip'})['shock_id']

		htmloutput = {
		'description' : description,
		'name' : htmloutput_name,
		'label' : 'Open New Tab',
		'shock_id': report_shock_id
		}

		if not for_predict:
			list_PickleFiles = os.listdir(os.path.join(self.scratch, 'forHTML', 'forDATA'))

			output_file_links = []

			for file in list_PickleFiles:
				output_file_links.append({'path' : os.path.join(self.scratch, 'forHTML', 'forDATA', file),
											'name' : file,
											'label': 'label' + str(file),
											'description': 'my_description'
											})

			"""output_zip_files.append({'path': os.path.join(read_file_path, file),
																					 'name': file,
																					 'label': label,
																					 'description': desc})"""

			report_params = {'message': '',
				 'workspace_name': current_ws,#params.get('input_ws'),
				 #'objects_created': objects_created,
				 'file_links': output_file_links,
				 'html_links': [htmloutput],
				 'direct_html_link_index': 0,
				 'html_window_height': 500,
				 'report_object_name': 'kb_classifier_report_' + str(uuid.uuid4())
				 }

		else:
			report_params = {'message': '',
				 'workspace_name': current_ws,#params.get('input_ws'),
				 #'objects_created': objects_created,
				 #'file_links': output_file_links,
				 'html_links': [htmloutput],
				 'direct_html_link_index': 0,
				 'html_window_height': 500,
				 'report_object_name': 'kb_classifier_report_' + str(uuid.uuid4())
				 }

		kbase_report_client = KBaseReport(self.callback_url, token=token)
		report_output = kbase_report_client.create_extended_report(report_params)

		output = {'report_name': report_output['name'], 'report_ref': report_output['ref']}

		print('I hope I am working now - this means that I am past the report generation')

		print(output.get('report_name')) # kb_classifier_report_5920d1da-2a99-463b-94a5-6cb8721fca45
		print(output.get('report_ref')) #19352/1/1

		return report_output

	##### Methods below are called only from inside this class ####

	def editBuildArguments(self, params):

		return_params = {
		"trainingset_name": params["trainingset_name"],
		"phenotypeclass": params["phenotypeclass"],
		"classifier": params.get("classifier"),
		"attribute": params["attribute"],
		"save_ts": params["save_ts"],
		"classifier_out": params["classifier_out"],
		"description" : params["description"] #edit this so classifier object can add this
		}

		print(return_params)

		if params.get("classifier") == "run_all":
			params["k_nearest_neighbors"] = {
			"n_neighbors": 5,
			"weights": "uniform",
			"algorithm": "auto",
			"leaf_size": 30,
			"p": 2,
			"metric": "minkowski",
			"metric_params": "",
			"knn_n_jobs": 1
			}
			return_params['k_nearest_neighbors'] = params["k_nearest_neighbors"]

			params["gaussian_nb"] = {
			"priors": ""
			}
			return_params['gaussian_nb'] = params["gaussian_nb"]

			params["logistic_regression"] = {
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
			}
			return_params['logistic_regression'] = params["logistic_regression"]

			params["decision_tree_classifier"] = {
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
			}
			return_params['decision_tree_classifier'] = params["decision_tree_classifier"]

			params["support_vector_machine"] = {
			"svm_C": 1,
			"kernel": "linear",
			"degree": 3,
			"gamma": "auto",
			"coef0": 0,
			"probability": "True",
			"shrinking": "True",
			"svm_tolerance": 0.001,
			"cache_size": 200,
			"svm_class_weight": "",
			"svm_verbose": "False",
			"svm_max_iter": -1,
			"decision_function_shape": "ovr",
			"svm_random_state": 0
			}
			return_params['support_vector_machine'] = params["support_vector_machine"]

			params["neural_network"] = {
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
			}
			return_params['neural_network'] = params["neural_network"]

		elif params.get("classifier") == "KNeighborsClassifier":
			if params["k_nearest_neighbors"] == None:
				params["k_nearest_neighbors"] = {
				"n_neighbors": 5,
				"weights": "uniform",
				"algorithm": "auto",
				"leaf_size": 30,
				"p": 2,
				"metric": "minkowski",
				"metric_params": "",
				"knn_n_jobs": 1
				}
			return_params['k_nearest_neighbors'] = params["k_nearest_neighbors"]

		elif params.get("classifier") == "GaussianNB":
			if params["gaussian_nb"] == None:
				params["gaussian_nb"] = {
				"priors": ""
				}
			return_params['gaussian_nb'] = params["gaussian_nb"]
		elif params.get("classifier") == "LogisticRegression":
			if params["logistic_regression"] == None:
				params["logistic_regression"] = {
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
				}
			return_params['logistic_regression'] = params["logistic_regression"]
		elif params.get("classifier") == "DecisionTreeClassifier":
			if params["decision_tree_classifier"] == None:
				params["decision_tree_classifier"] = {
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
				}
			return_params['decision_tree_classifier'] = params["decision_tree_classifier"]
		elif params.get("classifier") == "SVM":
			if params["support_vector_machine"] == None:
				params["support_vector_machine"] = {
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
				}
			return_params['support_vector_machine'] = params["support_vector_machine"]
		elif params.get("classifier") == "NeuralNetwork":
			if params["neural_network"] == None:
				params["neural_network"] = {
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
				}
			return_params['neural_network'] = params["neural_network"]

		if params["ensemble_model"] == None:
			params["ensemble_model"] = {
			"k_nearest_neighbors_box": 1,
			"gaussian_nb_box": 1,
			"logistic_regression_box": 1,
			"decision_tree_classifier_box": 1,
			"support_vector_machine_box": 1,
			"neural_network_box": 1,
			"voting": "soft",
			"en_weights": "",
			"en_n_jobs": 1,
			"flatten_transform": ""
			}
		return_params['ensemble_model'] = params["ensemble_model"]

		if params.get("classifier") == "KNeighborsClassifier":
			return_params['classifierAdvanced_params'] = params["k_nearest_neighbors"]

		if params.get("classifier") == "GaussianNB":
			return_params['classifierAdvanced_params'] = params["gaussian_nb"]

		if params.get("classifier") == "LogisticRegression":
			return_params['classifierAdvanced_params'] = params["logistic_regression"]
			
		if params.get("classifier") == "DecisionTreeClassifier":
			return_params['classifierAdvanced_params'] = params["decision_tree_classifier"]

		if params.get("classifier") == "SVM":
			return_params['classifierAdvanced_params'] = params["support_vector_machine"]

		if params.get("classifier") == "NeuralNetwork":
			return_params['classifierAdvanced_params'] = params["neural_network"]

		return return_params

	def incaseList_Names(self, list_name, for_predict = False):
		"""		
		args:
		---list_name (same as classifierTest)
		---for_predict is boolean indicator to see if this used by clf_Runner to pred_Runner
		does:
		---used when user inputs data/classifications in the textbox instead of excel file
		---creates a panda dataframe after parsing the input
		return:
		---the panda dataframe
		"""

		list_name = list_name.replace("\"", "")
		list_name = list_name.replace(", ", "\t")
		#list_name = list_name.replace("-", "\t")

		working_str = list_name.split("\\n")

		#open(u'kb/module/work/tmp/trialoutput.txt', u'w')
		tem_file = codecs.open(os.path.join(self.scratch, 'trialoutput.txt'), u"w", 'utf-8')
		for index in working_str:
			#print(index, file=tem_file)
			#print index
			print >>tem_file, index

		print ("before closing")

		tem_file.close()

		if not for_predict:
			print ("I'm inside the if not for_predict")
			my_workPD = pd.read_csv(os.path.join(self.scratch, 'trialoutput.txt'), dtype=str, delimiter="\s+")
			#print my_workPD
		else: 
			my_workPD = pd.read_csv(os.path.join(self.scratch, 'trialoutput.txt'),dtype=str)

		os.remove(os.path.join(self.scratch, 'trialoutput.txt'))

		return my_workPD

	# def createGenomeClassifierTrainingSet(self, current_ws, description, trainingset_object_Name, just_DF):
	# 	"""
	# 	args:
	# 	---current_ws is same as before
	# 	---trainingset_object_Name is the training set defined/input by the user
	# 	---just_DF is a dataframe that is given by the user in the form of an excel file or pasted in a text box, however converted in a data frame
	# 	does:
	# 	---takes the dataframe and pulls the Genome_ID and Classification and creates a trainingset_object (list of GenomeClass which holds Genome_ID and Classification
	# 	return:
	# 	---N/A just creates a trainingset_object in the workspace
	# 	"""
	# 	ctx = self.ctx

	# 	listintGNames = just_DF['Genome_ID']
		
	# 	#vigorous string matching izip(self.list_name, self.list_statistics)
	# 	listGNames = list(map(str, listintGNames))
	# 	for string, index in izip(listGNames, range(len(listGNames))):
	# 		listGNames[index] = string.replace(" ", "")

	# 	listClassification = just_DF['Classification']

	# 	list_GenomeClass = []
	# 	list_allGenomesinWS = []

	# 	missingGenomes = []

	# 	back = self.ws_client.list_objects({'workspaces':[current_ws],'type':'KBaseGenomes.Genome'})

	# 	print(back)

	# 	for item in back:
	# 		list_allGenomesinWS.append(item[1])

	# 	print(list_allGenomesinWS)

	# 	all_genome_ID = []
	# 	loaded_Narrative = []
	# 	all_Genome_Classification = []
	# 	add_trainingSet = []

	# 	for index in range(len(listGNames)):

	# 		try:
	# 			position = list_allGenomesinWS.index(listGNames[index])

	# 			print('printing positions')
	# 			print(position)

	# 			params_RAST =	{
	# 			"workspace": current_ws,#"sagoyal:narrative_1536939130038",
	# 			"input_genome": listGNames[index],
	# 			"output_genome": listGNames[index]+".RAST",
	# 			"call_features_rRNA_SEED": 0,
	# 			"call_features_tRNA_trnascan": 0,
	# 			"call_selenoproteins": 0,
	# 			"call_pyrrolysoproteins": 0,
	# 			"call_features_repeat_region_SEED": 0,
	# 			"call_features_strep_suis_repeat": 0,
	# 			"call_features_strep_pneumo_repeat": 0,
	# 			"call_features_crispr": 0,
	# 			"call_features_CDS_glimmer3": 0,
	# 			"call_features_CDS_prodigal": 0,
	# 			"annotate_proteins_kmer_v2": 1,
	# 			"kmer_v1_parameters": 1,
	# 			"annotate_proteins_similarity": 1,
	# 			"retain_old_anno_for_hypotheticals": 0,
	# 			"resolve_overlapping_features": 0,
	# 			"call_features_prophage_phispy": 0
	# 			}

	# 			output = self.rast.annotate_genome(params_RAST)

	# 			list_allGenomesinWSupdated = []

	# 			newBack = self.ws_client.list_objects({'workspaces':[current_ws],'type':'KBaseGenomes.Genome'})

	# 			for item in newBack:
	# 				list_allGenomesinWSupdated.append(item[1])

	# 			position = list_allGenomesinWSupdated.index(listGNames[index]+".RAST")

	# 			list_GenomeClass.append({'genome_ref': str(newBack[position][6]) + '/' + str(newBack[position][0]) + '/' + str(newBack[position][4]),#self.ws_client.get_objects([{'workspace':current_ws, 'name':listGNames[index]}])[0]['path'][0],
	# 										'genome_classification': listClassification[index],
	# 										'genome_name': listGNames[index]+".RAST",
	# 										'genome_id': 'my_genome_id',
	# 										'references': ['some','list'],
	# 										'evidence_types': ['another','some','list'],
	# 										})


	# 			# output = self.rast.annotate_genome(params_RAST)

	# 			# list_GenomeClass.append({'genome_ref': str(output[6]) + '/' + str(output[0]) + '/' + str(output[4]),#self.ws_client.get_objects([{'workspace':current_ws, 'name':listGNames[index]}])[0]['path'][0],
	# 			# 			'genome_classification': listClassification[index],
	# 			# 			'genome_name': listGNames[index]+".RAST",
	# 			# 			'genome_id': 'my_genome_id',
	# 			# 			'references': ['some','list'],
	# 			# 			'evidence_types': ['another','some','list'],
	# 			# 			})

	# 			# list_GenomeClass.append({'genome_ref': str(back[position][6]) + '/' + str(back[position][0]) + '/' + str(back[position][4]),#self.ws_client.get_objects([{'workspace':current_ws, 'name':listGNames[index]}])[0]['path'][0],
	# 			# 							'genome_classification': listClassification[index],
	# 			# 							'genome_name': listGNames[index],
	# 			# 							'genome_id': 'my_genome_id',
	# 			# 							'references': ['some','list'],
	# 			# 							'evidence_types': ['another','some','list'],
	# 			# 							})

	# 			all_genome_ID.append(listGNames[index])
	# 			loaded_Narrative.append(["Yes"])
	# 			all_Genome_Classification.append(listClassification[index])
	# 			add_trainingSet.append(["Yes"])
	# 		except:
	# 			print (listGNames[index])
	# 			print ('The above Genome does not exist in workspace')
	# 			missingGenomes.append(listGNames[index])

	# 			all_genome_ID.append(listGNames[index])
	# 			loaded_Narrative.append(["No"])
	# 			all_Genome_Classification.append(["None"])
	# 			add_trainingSet.append(["No"])
		
	# 	four_columns = pd.DataFrame.from_dict({'Genome Id': all_genome_ID, 'Loaded in the Narrative': loaded_Narrative, 'Classification' : all_Genome_Classification, 'Added to Training Set' : add_trainingSet})
	# 	four_columns = four_columns[['Genome Id', 'Loaded in the Narrative', 'Classification', 'Added to Training Set']]

	# 	old_width = pd.get_option('display.max_colwidth')
	# 	pd.set_option('display.max_colwidth', -1)
	# 	four_columns.to_html(os.path.join(self.scratch, 'forZeroHTML', 'four_columns.html'), index=False, justify='center')
	# 	pd.set_option('display.max_colwidth', old_width)

	# 	trainingset_object = {
	# 	'name': trainingset_object_Name,#'my_name',
	# 	'description': description,
	# 	'classification_type': 'my_classification_type',
	# 	'number_of_genomes': len(listGNames),
	# 	'number_of_classes': len(list(set(listClassification))),
	# 	'classes': list(set(listClassification)),
	# 	'classification_data': list_GenomeClass
	# 	}

	# 	obj_save_ref = self.ws_client.save_objects({'workspace': current_ws,
	# 												  'objects':[{
	# 												  'type': 'KBaseClassifier.GenomeClassifierTrainingSet',
	# 												  'data': trainingset_object,
	# 												  'name': trainingset_object_Name,  
	# 												  'provenance': ctx.get('provenance')  # ctx should be passed into this func.
	# 												  }]
	# 												})[0]

	# 	print "I'm print out the obj_save_ref"
	# 	print ""
	# 	print ""
	# 	print ""

	# 	print obj_save_ref
	# 	print "done"

	# 	return missingGenomes

	def createGenomeClassifierTrainingSet(self, current_ws, Annotated, just_DF, for_predict = False):
		"""
		args:
		---current_ws is same as before
		---trainingset_object_Name is the training set defined/input by the user
		---just_DF is a dataframe that is given by the user in the form of an excel file or pasted in a text box, however converted in a data frame
		does:
		---takes the dataframe and pulls the Genome_ID and Classification and creates a trainingset_object (list of GenomeClass which holds Genome_ID and Classification
		return:
		---N/A just creates a trainingset_object in the workspace
		"""
		ctx = self.ctx

		print("this is just_DF")
		print(just_DF)

		listintGNames = just_DF['Genome_ID']

		#vigorous string matching izip(self.list_name, self.list_statistics)
		listGNames = list(map(str, listintGNames))
		for string, index in izip(listGNames, range(len(listGNames))):
			listGNames[index] = string.replace(" ", "")

		
		inKBASE_Classification =[]

		list_allGenomesinWS = []

		missingGenomes = []

		back = self.ws_client.list_objects({'workspaces':[current_ws],'type':'KBaseGenomes.Genome'})

		#print(back)

		for item in back:
			list_allGenomesinWS.append(item[1])

		#print(list_allGenomesinWS)

		all_genome_ID = []
		loaded_Narrative = []
		all_Genome_Classification = []
		add_trainingSet = []

		inKBASE = []

		if(for_predict):
			for index in range(len(listGNames)):
				try:
					if(Annotated==1):
						position = list_allGenomesinWS.index(listGNames[index])
						inKBASE.append(listGNames[index])

					else:
						#do some rast annotation
						# position = list_allGenomesinWS.index(listGNames[index])
						# inKBASE.append(listGNames[index])

						for index in range(len(listGNames)):
							try:
								position = list_allGenomesinWS.index(listGNames[index])

								print('printing positions')
								print(position)

								params_RAST =	{
								"workspace": current_ws,#"sagoyal:narrative_1536939130038",
								"input_genome": listGNames[index],
								"output_genome": listGNames[index]+".RAST",
								"call_features_rRNA_SEED": 0,
								"call_features_tRNA_trnascan": 0,
								"call_selenoproteins": 0,
								"call_pyrrolysoproteins": 0,
								"call_features_repeat_region_SEED": 0,
								"call_features_strep_suis_repeat": 0,
								"call_features_strep_pneumo_repeat": 0,
								"call_features_crispr": 0,
								"call_features_CDS_glimmer3": 0,
								"call_features_CDS_prodigal": 0,
								"annotate_proteins_kmer_v2": 1,
								"kmer_v1_parameters": 1,
								"annotate_proteins_similarity": 1,
								"retain_old_anno_for_hypotheticals": 0,
								"resolve_overlapping_features": 0,
								"call_features_prophage_phispy": 0
								}

								output = self.rast.annotate_genome(params_RAST)

								inKBASE.append(listGNames[index]+".RAST")

							except:
								print (listGNames[index])
								print ('The above Genome does not exist in workspace')
								missingGenomes.append(listGNames[index])

				except:
					print (listGNames[index])
					print ('The above Genome does not exist in workspace')
					missingGenomes.append(listGNames[index])

			print(inKBASE)
			print(missingGenomes)

			return (missingGenomes, inKBASE)

		listClassification = just_DF['Classification']

		for index in range(len(listGNames)):

			try:
				position = list_allGenomesinWS.index(listGNames[index])

				print('printing positions')
				print(position)

				print("This is my Annotated value")
				print(Annotated)

				if(Annotated == 0):

					params_RAST =	{
					"workspace": current_ws,#"sagoyal:narrative_1536939130038",
					"input_genome": listGNames[index],
					"output_genome": listGNames[index]+".RAST",
					"call_features_rRNA_SEED": 0,
					"call_features_tRNA_trnascan": 0,
					"call_selenoproteins": 0,
					"call_pyrrolysoproteins": 0,
					"call_features_repeat_region_SEED": 0,
					"call_features_strep_suis_repeat": 0,
					"call_features_strep_pneumo_repeat": 0,
					"call_features_crispr": 0,
					"call_features_CDS_glimmer3": 0,
					"call_features_CDS_prodigal": 0,
					"annotate_proteins_kmer_v2": 1,
					"kmer_v1_parameters": 1,
					"annotate_proteins_similarity": 1,
					"retain_old_anno_for_hypotheticals": 0,
					"resolve_overlapping_features": 0,
					"call_features_prophage_phispy": 0
					}

					output = self.rast.annotate_genome(params_RAST)

					inKBASE.append(listGNames[index]+".RAST")
					inKBASE_Classification.append(listClassification[index])

					all_genome_ID.append(listGNames[index]+".RAST")
					loaded_Narrative.append(["Yes"])
					all_Genome_Classification.append(listClassification[index])
					add_trainingSet.append(["Yes"])

				else:
					# you will end up with case where the genomes will be RAST annotated but not have .RAST attached to it

					inKBASE.append(listGNames[index])
					inKBASE_Classification.append(listClassification[index])

					all_genome_ID.append(listGNames[index])
					loaded_Narrative.append(["Yes"])
					all_Genome_Classification.append(listClassification[index])
					add_trainingSet.append(["Yes"])

			except:
				print (listGNames[index])
				print ('The above Genome does not exist in workspace')
				missingGenomes.append(listGNames[index])

				all_genome_ID.append(listGNames[index])
				loaded_Narrative.append(["No"])
				all_Genome_Classification.append(["None"])
				add_trainingSet.append(["No"])
		
		four_columns = pd.DataFrame.from_dict({'Genome Id': all_genome_ID, 'Loaded in the Narrative': loaded_Narrative, 'Classification' : all_Genome_Classification, 'Added to Training Set' : add_trainingSet})
		four_columns = four_columns[['Genome Id', 'Loaded in the Narrative', 'Classification', 'Added to Training Set']]

		old_width = pd.get_option('display.max_colwidth')
		pd.set_option('display.max_colwidth', -1)
		four_columns.to_html(os.path.join(self.scratch, 'forZeroHTML', 'four_columns.html'), index=False, justify='center', table_id = "four_columns", classes =["table", "table-striped", "table-bordered"])
		pd.set_option('display.max_colwidth', old_width)


		print "I'm print out the obj_save_ref"
		print ""
		print ""
		print ""

		#print obj_save_ref
		print "done"

		return (missingGenomes, inKBASE, inKBASE_Classification)

	def newReferencetoGenome(self, current_ws, description, trainingset_object_Name, inKBASE, inKBASE_Classification):

		ctx = self.ctx

		list_GenomeClass = []
		list_allGenomesinWSupdated = []

		newBack = self.ws_client.list_objects({'workspaces':[current_ws],'type':'KBaseGenomes.Genome'})

		for item in newBack:
			list_allGenomesinWSupdated.append(item[1])

		for index in range(len(inKBASE)):
			position = list_allGenomesinWSupdated.index(inKBASE[index])

			list_GenomeClass.append({'genome_ref': str(newBack[position][6]) + '/' + str(newBack[position][0]) + '/' + str(newBack[position][4]),#self.ws_client.get_objects([{'workspace':current_ws, 'name':listGNames[index]}])[0]['path'][0],
										'genome_classification': inKBASE_Classification[index],
										'genome_name': inKBASE[index],
										'genome_id': 'my_genome_id',
										'references': ['some','list'],
										'evidence_types': ['another','some','list'],
										})


		trainingset_object = {
			'name': trainingset_object_Name,#'my_name',
			'description': description,
			'classification_type': 'my_classification_type',
			'number_of_genomes': len(inKBASE),
			'number_of_classes': len(list(set(inKBASE_Classification))),
			'classes': list(set(inKBASE_Classification)),
			'classification_data': list_GenomeClass
			}

		obj_save_ref = self.ws_client.save_objects({'workspace': current_ws,
													  'objects':[{
													  'type': 'KBaseClassifier.GenomeClassifierTrainingSet',
													  'data': trainingset_object,
													  'name': trainingset_object_Name,  
													  'provenance': ctx.get('provenance')  # ctx should be passed into this func.
													  }]
													})[0]

	def unloadGenomeClassifierTrainingSet(self, current_ws, trainingset_name):
		"""
		args:
		---current_ws is same as before
		---trainingset_name is the training set selected by the user
		does:
		---from the training set object it extracts the Genome_ID and Classification and creates a dataframe of them
		return:
		---the dataframe
		"""

		input_trainingset_object = self.ws_client.get_objects([{'workspace':current_ws, 'name':trainingset_name}])
		trainingset_object = input_trainingset_object[0]['data']['classification_data']
		training_set_ref = input_trainingset_object[0]['path'][0]

		iterations = len(trainingset_object)

		listGNames = [] #just_DF['Genome_ID']
		listrefs = []
		listClassification = [] #just_DF['Classification']

		for example in range(iterations):
			print(trainingset_object[example]['genome_name'])
			listGNames.append(trainingset_object[example]['genome_name'])
			listrefs.append(trainingset_object[example]['genome_ref'])
			listClassification.append(trainingset_object[example]['genome_classification'])

		print(listGNames)
		print(listClassification)

		detailsDF = {'Genome_ID': listGNames,
					'Classification': listClassification
					}

		numClasses = len(set(listClassification))

		remadeDF = pd.DataFrame.from_dict(detailsDF)

		return remadeDF, training_set_ref, numClasses, listrefs

	def intake_method(self, just_DF, for_predict = False):
		"""
		args:
		---just_DF is pandas dataframe made from incaseList_Names or from shock--> converted to pd
		---for_predict is boolean indicator to see if this used by clf_Runner to pred_Runner
		does:
		---with the excel file which has 2 columns: Genome_ID (same as my_input) and Classification
			---it creates another dataframe with only classifications and rows as "index" which are genome names (my_input)
		return:
		---my_all_classifications the dataframe with all_classifications (essentially the Y variable for ML)
		---my_all_classifications.index the index or 'rows' as list (listOfNames) of my_all_classifications
		"""

		my_all_classifications = just_DF

		#print my_all_classifications
		print my_all_classifications.columns.values.tolist()

		my_all_classifications.set_index('Genome_ID', inplace=True)
		#my_all_classifications.reset_index()
		#my_all_classifications.set_index(list(my_all_classifications.index),inplace=True)

		print "Below is my_all_classifications"

		#print my_all_classifications

		if not for_predict:
			return list(my_all_classifications.index), my_all_classifications
		else:
			return list(my_all_classifications.index)


	def get_wholeClassification(self, listOfNames, current_ws, search_attribute, refs = None, master_Role = None, for_predict = False):
		"""
		args:
		---listOfNames is a list from the dataframe.index containing the 'rows' which is names/Genome_ID
		---current_ws (same as before)
		---master_Role is given if being called from predict_phenotype method otherwise not
		---for_predict is boolean indicator to see if this used by clf_Runner to pred_Runner
		does:
		---creates a dataframe for the all the genomes given
			---Rows are "index" which is the name of the genome(same as my_input)
			---Colmuns are "master Role" which is a list of the all functional roles
		return:
		---returns the dataframe which contains all_attributes (this is the X matrix for ML)
		"""

		print current_ws

		if not for_predict:
			master_Role = [] #make this master_Role


		# functionList = self.ws_client.get_objects([{'workspace':current_ws, 'name':"679190.3.RAST"}])[0]['data']['non_coding_features']

		# print("here is functionList")
		# print( listOfNames)

		name_and_roles = {}

		search = ""
		if (search_attribute == "functional_roles"):
			search = 'function'
		elif (search_attribute == "prot_seq"):
			search = 'protein_translation'


		# for current_gName in listOfNames:
		for current_ref, current_gName in izip(refs, listOfNames):
			listOfFunctionalRoles = []
			try:
				#functionList = self.ws_client.get_objects([{'workspace':current_ws, 'name':current_gName}])[0]['data']['cdss']
				#functionList = self.ws_client.get_objects([{'workspace':current_ws, 'name':current_gName}])[0]['data']['non_coding_features']
				functionList = self.ws_client.get_objects2({'objects' : [{'ref' : current_ref}]})['data'][0]['data']['non_coding_features']

				print("here is functionList")

				for function in range(len (functionList)):
					if str(functionList[function][search][0]).lower() != 'hypothetical protein':
						# print("functionList[function][search]")
						# print(functionList[function][search])
						#print(str(functionList[function]['functions'][0]).find(" @ " ))
						#if (str(functionList[function]['functions'][0]).find(" @ " ) > 0):
						if " @ " in str(functionList[function][search][0]):
							listOfFunctionalRoles.extend(str(functionList[function][search][0]).split(" @ "))
							print("I went inside the if statement")
						elif " / " in str(functionList[function][search][0]):
							listOfFunctionalRoles.extend(str(functionList[function][search][0]).split(" / "))
						elif "; " in str(functionList[function][search][0]):
							listOfFunctionalRoles.extend(str(functionList[function][search][0]).split("; "))
						else:
							listOfFunctionalRoles.append(str(functionList[function][search]))

				# for function in range(len (functionList)):
				# 	if str(functionList[function]['functions'][0]).lower() != 'hypothetical protein':
				# 		#print(str(functionList[function]['functions'][0]).find(" @ " ))
				# 		#if (str(functionList[function]['functions'][0]).find(" @ " ) > 0):
				# 		if " @ " in str(functionList[function]['functions'][0]):
				# 			listOfFunctionalRoles.extend(str(functionList[function]['functions'][0]).split(" @ "))
				# 			print("I went inside the if statement")
				# 		elif " / " in str(functionList[function]['functions'][0]):
				# 			listOfFunctionalRoles.extend(str(functionList[function]['functions'][0]).split(" / "))
				# 		elif "; " in str(functionList[function]['functions'][0]):
				# 			listOfFunctionalRoles.extend(str(functionList[function]['functions'][0]).split("; "))
				# 		else:
				# 			listOfFunctionalRoles.append(str(functionList[function]['functions'][0]))

			except:
				functionList = self.ws_client.get_objects2({'objects' : [{'ref' : current_ref}]})['data'][0]['data']['features']
				#['data'][0]['data']['features']
				#functionList = self.ws_client.get_objects([{'workspace':current_ws, 'name':current_gName}])[0]['data']['features']
				# functionList = self.ws_client.get_objects([{'workspace':current_ws, 'name':current_gName}])[0]['data']['non_coding_features']


				# print("here is functionList")
				# print(functionList)

				for function in range(len (functionList)):
					if str(functionList[function][search]).lower() != 'hypothetical protein':
						#print(str(functionList[function]['functions'][0]).find(" @ " ))
						#if (str(functionList[function]['functions'][0]).find(" @ " ) > 0):
						if " @ " in str(functionList[function][search]):
							listOfFunctionalRoles.extend(str(functionList[function][search]).split(" @ "))
							print("I went inside the if statement #2")
						elif " / " in str(functionList[function][search]):
							listOfFunctionalRoles.extend(str(functionList[function][search]).split(" / "))
						elif "; " in str(functionList[function][search]):
							listOfFunctionalRoles.extend(str(functionList[function][search]).split("; "))
						else:
							listOfFunctionalRoles.append(str(functionList[function][search]))


			name_and_roles[current_gName] = listOfFunctionalRoles

			print "I have arrived inside the desired for loop!!"
			print(len(listOfFunctionalRoles))
			print(current_gName)

		if not for_predict:
			master_pre_Role = list(itertools.chain(*name_and_roles.values()))
			master_Role = list(set(master_pre_Role))
		

		print("this is my master_Role")
		# print(master_Role)

		#In case you want to save functional roles (master_Role) and the dictionary containing {Genome_ID: [Functional Roles]}
		"""
		with open(os.path.join(self.scratch, "KBASEfunctionalRoles.txt"), "w") as f:
			f.write(unicode(str(master_Role)))

		import json

		my_json = json.dumps(name_and_roles)
		with open(os.path.join(self.scratch,"KBASEname_and_roles.json"),"w") as f:
			f.write(unicode(my_json))
		"""
		
		data_dict = {}

		for current_gName in listOfNames:
			arrayofONEZERO = []

			current_Roles = name_and_roles[current_gName]

			for individual_role in master_Role:
				if individual_role in current_Roles:
					arrayofONEZERO.append(1)
				else:
					arrayofONEZERO.append(0)

			data_dict[current_gName] = arrayofONEZERO

		my_all_attributes = pd.DataFrame.from_dict(data_dict, orient='index', columns = master_Role)

		#pickle_out = open(os.path.join(self.scratch,"fromKBASE_Phylum_attributes.pickle"), "wb")
		#pickle.dump(my_all_attributes, pickle_out)

		#pickle_out = open(os.path.join(self.scratch,"fromKBASE_Phylum_MR.pickle"), "wb")
		#pickle.dump(master_Role, pickle_out)

		print("I'm done creating the all_attributes data frame")

		if not for_predict:
			return my_all_attributes, master_Role
		else:
			return my_all_attributes
		

	def createHoldingDirs(self, clf = True):
		"""
		does:
		---creates the directories that hold all of the html files, figures, and .pickle files
		"""

		if clf:
			dirs_names = ['pics', 'dotFolder', 'forHTML', 'forHTML/html1folder', 'forHTML/html2folder', 'forHTML/html4folder',  'forHTML/forDATA']
		else:	#pred
			dirs_names = ['forSecHTML', 'forSecHTML/html3folder']

		for name in dirs_names:
			out_path = os.path.join(self.scratch, name)
			os.makedirs(out_path)

	def whichClassifier(self, name):
		"""
		args:
		---name which is a string that the user will pass in as to which classifier (sklearn) classifier they want
		does:
		---matches string with sklearn classifier
		return:
		---sklearn classifier
		"""

		if name == u"KNeighborsClassifier":
			return KNeighborsClassifier()
		elif name == u"GaussianNB":
			return GaussianNB()
		elif name == u"LogisticRegression":
			return LogisticRegression(random_state=0)
		elif name == u"DecisionTreeClassifier":
			return DecisionTreeClassifier(random_state=0)
		elif name == u"SVM":
			return svm.SVC(kernel = "linear",random_state=0)
		elif name == u"NeuralNetwork":
			return MLPClassifier(random_state=0)
		else:
			return u"ERROR THIS SHOULD NOT HAVE REACHED HERE"

	def whichClassifierAdvanced(self, name, clfA_params, multi_call = False):
		"""
		args:
		---name which is a string that the user will pass in as to which classifier (sklearn) classifier they want
		does:
		---matches string with sklearn classifier
		return:
		---sklearn classifier
		"""

		if name == u"KNeighborsClassifier":
			print("here calling clfA_params")
			print(type(clfA_params))
			clfA_params = self.fixKNN(clfA_params, multi_call)
			return KNeighborsClassifier(n_neighbors=clfA_params["n_neighbors"], weights=clfA_params["weights"], algorithm=clfA_params["algorithm"], leaf_size=clfA_params["leaf_size"], p=2, metric=clfA_params["metric"], metric_params=clfA_params["metric_params"], n_jobs=clfA_params["knn_n_jobs"])
		
		elif name == u"GaussianNB":
			clfA_params = self.fixGNB(clfA_params, multi_call)
			return GaussianNB(priors=clfA_params["priors"])
		
		elif name == u"LogisticRegression":
			clfA_params = self.fixLR(clfA_params, multi_call)
			return LogisticRegression(penalty=clfA_params["penalty"], dual=clfA_params["dual"], tol=clfA_params["lr_tolerance"], C=clfA_params["lr_C"], fit_intercept=clfA_params["fit_intercept"], intercept_scaling=clfA_params["intercept_scaling"], class_weight=clfA_params["lr_class_weight"], random_state=clfA_params["lr_random_state"], solver=clfA_params["lr_solver"], max_iter=clfA_params["lr_max_iter"], multi_class=clfA_params["multi_class"], verbose=clfA_params["lr_verbose"], warm_start=clfA_params["lr_warm_start"], n_jobs=clfA_params["lr_n_jobs"])
		
		elif name == u"DecisionTreeClassifier":
			clfA_params = self.fixDTC(clfA_params, multi_call)
			return DecisionTreeClassifier(criterion=clfA_params["criterion"], splitter=clfA_params["splitter"], max_depth=clfA_params["max_depth"], min_samples_split=clfA_params["min_samples_split"], min_samples_leaf=clfA_params["min_samples_leaf"], min_weight_fraction_leaf=clfA_params["min_weight_fraction_leaf"], max_features=clfA_params["max_features"], random_state=clfA_params["dt_random_state"], max_leaf_nodes=clfA_params["max_leaf_nodes"], min_impurity_decrease=clfA_params["min_impurity_decrease"], class_weight= clfA_params["dt_class_weight"], presort=clfA_params["presort"])
		
		elif name == u"SVM":
			clfA_params = self.fixSVM(clfA_params, multi_call)
			return svm.SVC(C=clfA_params["svm_C"], kernel=clfA_params["kernel"], degree=clfA_params["degree"], gamma=clfA_params["gamma"], coef0=clfA_params["coef0"], shrinking=clfA_params["shrinking"], probability=clfA_params["probability"], tol=clfA_params["svm_tolerance"], cache_size=clfA_params["cache_size"], class_weight=clfA_params["svm_class_weight"], verbose=clfA_params["svm_verbose"], max_iter=clfA_params["svm_max_iter"], decision_function_shape=clfA_params["decision_function_shape"], random_state=clfA_params["svm_random_state"])
		
		elif name == u"NeuralNetwork":
			clfA_params = self.fixNN(clfA_params, multi_call)
			return MLPClassifier(hidden_layer_sizes=clfA_params["hidden_layer_sizes"], activation=clfA_params["activation"], solver=clfA_params["mlp_solver"], alpha=clfA_params["alpha"], batch_size=clfA_params["batch_size"], learning_rate=clfA_params["learning_rate"], learning_rate_init=clfA_params["learning_rate_init"], power_t=clfA_params["power_t"], max_iter=clfA_params["mlp_max_iter"], shuffle=clfA_params["shuffle"], random_state=clfA_params["mlp_random_state"], tol=clfA_params["mlp_tolerance"], verbose=clfA_params["mlp_verbose"], warm_start=clfA_params["mlp_warm_start"], momentum=clfA_params["momentum"], nesterovs_momentum=clfA_params["nesterovs_momentum"], early_stopping=clfA_params["early_stopping"], validation_fraction=clfA_params["validation_fraction"], beta_1=clfA_params["beta_1"], beta_2=clfA_params["beta_2"], epsilon=clfA_params["epsilon"])
		
		else:
			return u"ERROR THIS SHOULD NOT HAVE REACHED HERE"

	def ensembleCreation(self, ensemble_params, params):

		my_estimators = []

		estimators_inHTML = ""

		if ensemble_params["k_nearest_neighbors_box"] == 1:
			my_estimators.extend([("knn",self.whichClassifierAdvanced("KNeighborsClassifier", params["k_nearest_neighbors"], multi_call = True))])
			estimators_inHTML += "K Nearest Neighbors Classifier, "
		if ensemble_params["gaussian_nb_box"] == 1:
			my_estimators.extend([("gnb",self.whichClassifierAdvanced("GaussianNB", params["gaussian_nb"], multi_call = True))])
			estimators_inHTML += "Gaussian Naive Bayes Classifier, "
		if ensemble_params["logistic_regression_box"] == 1:
			my_estimators.extend([("lr",self.whichClassifierAdvanced("LogisticRegression", params["logistic_regression"],multi_call = True))])
			estimators_inHTML += "Logistic Regression Classifier, "
		if ensemble_params["decision_tree_classifier_box"] == 1:
			my_estimators.extend([("dtc",self.whichClassifierAdvanced("DecisionTreeClassifier", params["decision_tree_classifier"], multi_call = True))])
			estimators_inHTML += "Decision Tree Classifier, "
		if ensemble_params["support_vector_machine_box"] == 1:
			my_estimators.extend([("svm",self.whichClassifierAdvanced("SVM", params["support_vector_machine"], multi_call = True))])
			estimators_inHTML += "Support Vector Machine, "
		if ensemble_params["neural_network_box"] == 1:
			my_estimators.extend([("nn",self.whichClassifierAdvanced("NeuralNetwork", params["neural_network"], multi_call = True))])
			estimators_inHTML += "Neural Network"

		ensemble_params = self.fixEnsemble(ensemble_params)

		print("Here are my estimators")
		#print(my_estimators)

		print("Here are my ensemble_params")
		#print(ensemble_params)

		if estimators_inHTML == "":
			return "No_Third", ""
		else:
			return VotingClassifier(estimators=my_estimators, voting=ensemble_params["voting"], n_jobs=ensemble_params["en_n_jobs"], flatten_transform=ensemble_params["flatten_transform"]), estimators_inHTML

	def fixKNN(self, clfA_params, round_best):

		if not round_best:
			#convert string to dictionary
			if clfA_params["metric_params"] == "":
				clfA_params["metric_params"] = None
			else:
				clfA_params["metric_params"] = ast.literal_eval(clfA_params["metric_params"])

			return clfA_params

		else:
			return clfA_params

	def fixGNB(self, clfA_params, round_best):

		if not round_best:
			#convert string to list
			if clfA_params["priors"] == "":
				clfA_params["priors"] = None
			else:
				clfA_params["priors"] = ast.literal_eval(clfA_params["priors"])

			return clfA_params

		else:
			return clfA_params

	def fixLR(self, clfA_params, round_best):

		if not round_best:
			clfA_params["dual"] = self.str_to_bool(clfA_params["dual"])
			clfA_params["fit_intercept"] = self.str_to_bool(clfA_params["fit_intercept"])

			if clfA_params["lr_class_weight"] == "":
				clfA_params["lr_class_weight"] = None
			elif clfA_params["lr_class_weight"] == "balanced":
				pass
			else:
				clfA_params["lr_class_weight"] = ast.literal_eval(clfA_params["lr_class_weight"])

			clfA_params["lr_warm_start"] = self.str_to_bool(clfA_params["lr_warm_start"])

			return clfA_params

		else:
			return clfA_params

	def fixDTC(self, clfA_params, round_best):

		if not round_best:
			# check if int, float, or None
			clfA_params["presort"] = self.str_to_bool(clfA_params["presort"])
			if clfA_params["dt_class_weight"] == "":
				clfA_params["dt_class_weight"] = None
			elif clfA_params["dt_class_weight"] == "balanced":
				pass
			else:
				clfA_params["dt_class_weight"] = ast.literal_eval(clfA_params["dt_class_weight"])


			if clfA_params["max_features"] == "":
				clfA_params["max_features"] = None
			else:
				pass

			try:
				int(clfA_params["max_features"])
				clfA_params["max_features"] == int(clfA_params["max_features"])
			except:
				pass

			try:
				float(clfA_params["max_features"])
				clfA_params["max_features"] == float(clfA_params["max_features"])
			except:
				pass

			print ("My current clfA is:")
			print(clfA_params)

			return clfA_params

		else:
			return clfA_params

	def fixSVM(self, clfA_params, round_best):

		if not round_best:
			clfA_params["probability"] = self.str_to_bool(clfA_params["probability"])
			clfA_params["shrinking"] = self.str_to_bool(clfA_params["shrinking"])

			if clfA_params["svm_class_weight"] == "":
				clfA_params["svm_class_weight"] = None
			elif clfA_params["svm_class_weight"] == "balanced":
				pass
			else:
				clfA_params["svm_class_weight"] = ast.literal_eval(clfA_params["svm_class_weight"])

			clfA_params["svm_verbose"] = self.str_to_bool(clfA_params["svm_verbose"])

			return clfA_params

		else:
			return clfA_params

	def fixNN(self, clfA_params, round_best):

		if not round_best:
			#convert string to tuple
			clfA_params["hidden_layer_sizes"] = ast.literal_eval(clfA_params["hidden_layer_sizes"])

			clfA_params["shuffle"] = self.str_to_bool(clfA_params["shuffle"])
			clfA_params["mlp_verbose"] = self.str_to_bool(clfA_params["mlp_verbose"])
			clfA_params["mlp_warm_start"] = self.str_to_bool(clfA_params["mlp_warm_start"])
			clfA_params["nesterovs_momentum"] = self.str_to_bool(clfA_params["nesterovs_momentum"])
			clfA_params["early_stopping"] = self.str_to_bool(clfA_params["early_stopping"])

			return clfA_params

		else:
			return clfA_params

	def fixEnsemble(self, ensemble_params):
		
		if ensemble_params["en_weights"] == "":
				ensemble_params["en_weights"] = None
		else:
			ensemble_params["en_weights"] = ast.literal_eval(ensemble_params["en_weights"])

		if ensemble_params["flatten_transform"] == "":
				ensemble_params["flatten_transform"] = None
		else:
			ensemble_params["flatten_transform"] = self.str_to_bool(ensemble_params["flatten_transform"])

		return ensemble_params

	def str_to_bool(self, s):
		#Convert string to Boolean
		if s == 'True':
			 return True
		elif s == 'False':
			 return False

	def classifierTest(self, classifierTest_params):
		"""
		args:
		---classifierTest_params dictionary containing:
			'ctx' - context variable from narrative
			'current_ws' - current_ws which is a narrative enviornment variable necessary to access the KBase workspace
			'classifier' - classifier which is a sklearn object that has methods #LogisticRegression()
			'classifier_type' - classifier in string format
			'classifier_name' - string and is what is the name given by user to what the classifer is being saved as
			'my_mappng' - dictionary that converts the classifications to nums ie. ['A','B', 'C'] --> [0,1,2]
			'master_Role' - column names of all_attributes which is same as all of the functional roles
			'splits' - number SKF splits
			'train_index' - array of training sets
			'test_index' - array of testing sets
			'all_attributes' - the X's in 1s and 0s
			'all_classifications' - the Y's classification
			'class_list' - the 'keys' in the my_mapping ['A','B', 'C']
			'htmlfolder' - name of folder that saves the classifier_object
			'print_cfm - print_cfm is boolean (False when running through tuning and you don't want to print out all results on the
								console - True otherwise)
			}

		does:
		---calculates the numerical value of the the classifiers
		---saves down pickled versions of classifiers (probably make a separate method)
			---saves down base64 versions of classifiers
		---creates the text fields and values to be placed into the statistics table
		---calls the plot_confusion_matrix function

		return:
		--- (np.average(train_score), np.std(train_score), np.average(validate_score), np.std(validate_score))
			---return statement is only used when you repeatedly loop through this function during tuning
		"""

		ctx = classifierTest_params['ctx']
		current_ws = classifierTest_params['current_ws']
		classifier = classifierTest_params['classifier']
		classifier_type = classifierTest_params['classifier_type']
		classifier_name = classifierTest_params['classifier_name']
		my_mapping = classifierTest_params['my_mapping']
		master_Role = classifierTest_params['master_Role']

		splits = classifierTest_params['splits']
		train_index = classifierTest_params['train_index']
		test_index = classifierTest_params['test_index']

		all_attributes = classifierTest_params['all_attributes']
		all_classifications = classifierTest_params['all_classifications']
		class_list = classifierTest_params['class_list']
		htmlfolder = classifierTest_params['htmlfolder']
		print_cfm = classifierTest_params['print_cfm']
		training_set_ref = classifierTest_params['training_set_ref']
		description = classifierTest_params['description']

		if print_cfm:
			print classifier_name
			self.list_name.extend([classifier_name])
		
		train_score = np.zeros(splits)
		validate_score = np.zeros(splits)
		matrix_size = class_list.__len__()

		cnf_matrix = np.zeros(shape=(matrix_size, matrix_size))
		cnf_matrix_f = np.zeros(shape=(matrix_size, matrix_size))
		
		for c in range(splits):
			X_train = all_attributes[train_index[c]]
			y_train = all_classifications[train_index[c]]
			X_test = all_attributes[test_index[c]]
			y_test = all_classifications[test_index[c]]
			classifier.fit(X_train, y_train)
			train_score[c] = classifier.score(X_train, y_train)
			validate_score[c] = classifier.score(X_test, y_test)
			y_pred = classifier.predict(X_test)
			cnf = confusion_matrix(y_test, y_pred)
			cnf_f = cnf.astype(u'float') / cnf.sum(axis=1)[:, np.newaxis]
			for i in range(len(cnf)):
				for j in range(len(cnf)):
					cnf_matrix[i][j] += cnf[i][j]
					cnf_matrix_f[i][j] += cnf_f[i][j]

		if print_cfm:
			print("I'm inside this print_cfm")
			pickle_out = open(os.path.join(self.scratch, 'forHTML', 'forDATA', unicode(classifier_name) + u".pickle"), u"wb")

			#pickle_out = open("/kb/module/work/tmp/" + str(self.classifier_name) + ".pickle", "wb")


			pickle.dump(classifier.fit(all_attributes, all_classifications), pickle_out, protocol = 2)
			pickle_out.close()
			print("I've dumped and saved the pickle file")

			#just temporary trial thing
			"""
			Pickle_folder = os.path.join('savingPickle')
			os.mkdir(Pickle_folder)

			pickle_out = open(os.path.join(Pickle_folder, unicode(classifier_name) + u".pickle"), u"wb")
			pickle.dump(classifier.fit(all_attributes, all_classifications), pickle_out, protocol = 2)
			pickle_out.close()

			"""
			
			print("trying to save shock stuff")
			shock_id, handle_id = self._upload_to_shock(os.path.join(self.scratch, 'forHTML', 'forDATA', unicode(classifier_name) + u".pickle"))
			

			#handle_id = 'will fix later'
			
			#base64
			#current_pickle = pickle.dumps(classifier.fit(all_attributes, all_classifications), protocol=0)
			#pickled = codecs.encode(current_pickle, "base64").decode()

			pickled = "this is what the pickled string would be"

			print ""
			print "This is printing out the classifier_object that needs to be saved down dump"
			print ""

			print "your training_set_ref is below"
			print(training_set_ref)

			
			classifier_object = {
			'classifier_id' : '',
			'classifier_type' : classifier_type, # Neural network
			'classifier_name' : classifier_name,
			#'classifier_data' : pickled,
			'classifier_handle_ref' : handle_id,
			'classifier_description' : description,
			'lib_name' : 'sklearn',
			'attribute_type' : 'functional_roles',
			'number_of_attributes' : all_attributes.shape[1],#class_list.__len__(),
			'attribute_data' : master_Role,#["this is where master_role would go", "just a list"],#master_Role, #master_Role,
			'class_list_mapping' : my_mapping, #{} my_mapping, #my_mapping,
			'number_of_genomes' : class_list.__len__()#, #all_attributes.shape[1],
			#'training_set_ref' : training_set_ref #self.dfu.get_objects({'object_refs': [training_set_ref]}) #training_set_ref
			}

			if training_set_ref != 'User Denied':
				classifier_object['training_set_ref'] = training_set_ref
			#print classifier_object

			#Saving the Classifier object
	
			obj_save_ref = self.ws_client.save_objects({'workspace': current_ws,
														  'objects':[{
														  'type': 'KBaseClassifier.GenomeCategorizer',
														  'data': classifier_object,
														  'name': classifier_name,  
														  'provenance': ctx.get('provenance')  # ctx should be passed into this func.
														  }]
														})[0]

			print "I'm print out the obj_save_ref"
			print ""
			print ""
			print ""

			print obj_save_ref
			print "done"        

		if print_cfm:

			cnf_av = cnf_matrix/splits
			self.NClasses(class_list, cnf_av)

			self.plot_confusion_matrix(np.round(cnf_matrix_f/splits*100.0,1),class_list,u'Confusion Matrix', htmlfolder, classifier_name, classifier_type)

		"""
		list_forDict = []

		if class_list.__len__() == 3:
			if print_cfm:
				cnf_av = cnf_matrix / splits
				print
				print cnf_av[0][0], cnf_av[0][1], cnf_av[0][2]
				print cnf_av[1][0], cnf_av[1][1], cnf_av[1][2]
				print cnf_av[2][0], cnf_av[2][1], cnf_av[2][2]
				print
				print class_list[0]
				TP = cnf_av[0][0]
				TN = cnf_av[1][2] + cnf_av[1][2] + cnf_av[2][1] + cnf_av[2][2]
				FP = cnf_av[0][1] + cnf_av[0][2]
				FN = cnf_av[1][0] + cnf_av[2][0]
				list_forDict.extend([None])
				list_forDict.extend(self.cf_stats(TN, TP, FP, FN))

				print class_list[1]
				TP = cnf_av[1][1]
				TN = cnf_av[0][0] + cnf_av[0][2] + cnf_av[2][0] + cnf_av[2][2]
				FP = cnf_av[1][0] + cnf_av[1][2]
				FN = cnf_av[0][1] + cnf_av[2][1]
				list_forDict.extend([None, None])
				list_forDict.extend(self.cf_stats(TN, TP, FP, FN))

				print class_list[2]
				TP = cnf_av[2][2]
				TN = cnf_av[0][0] + cnf_av[0][1] + cnf_av[1][0] + cnf_av[1][1]
				FP = cnf_av[2][0] + cnf_av[2][1]
				FN = cnf_av[0][1] + cnf_av[0][2]
				list_forDict.extend([None, None])
				list_forDict.extend(self.cf_stats(TN, TP, FP, FN))

				list_forDict.extend([(list_forDict[4] + list_forDict[10] + list_forDict[16])/3])

				self.list_statistics.append(list_forDict)

				# self.plot_confusion_matrix(cnf_matrix/10,class_list,'Confusion Matrix')
				self.plot_confusion_matrix(cnf_matrix_f/splits*100.0,class_list,u'Confusion Matrix', htmlfolder, classifier_name, classifier_type)

		if class_list.__len__() == 2:
			if print_cfm:

				TP = cnf[0][0]
				TN = cnf[1][1]
				FP = cnf[0][1]
				FN = cnf[1][0]

				list_forDict.extend(self.cf_stats(TN, TP, FP, FN))
				self.list_statistics.append(list_forDict)

				self.plot_confusion_matrix(cnf_matrix_f/splits*100.0,class_list,u'Confusion Matrix', htmlfolder, classifier_name, classifier_type)
		"""

		if print_cfm:
			print classifier
			print
			print u"Confusion matrix"
			for i in xrange(len(cnf_matrix)):
				print class_list[i],; sys.stdout.write(u"  \t")
				for j in xrange(len(cnf_matrix[i])):
					print cnf_matrix[i][j] / splits,; sys.stdout.write(u"\t")
				print
			print
			for i in xrange(len(cnf_matrix_f)):
				print class_list[i],; sys.stdout.write(u"  \t")
				for j in xrange(len(cnf_matrix_f[i])):
					print u"%6.1f" % ((cnf_matrix_f[i][j] / splits) * 100.0),; sys.stdout.write(u"\t")
				print
			print
			print u"01", cnf_matrix[0][1]

		print u"%6.3f\t%6.3f\t%6.3f\t%6.3f" % (
		np.average(train_score), np.std(train_score), np.average(validate_score), np.std(validate_score))

		return (np.average(train_score), np.std(train_score), np.average(validate_score), np.std(validate_score))

	def NClasses(self, class_list, cnf_av):
		
		list_forDict = []

		for class_current in range(len(class_list)):
			print(class_list[class_current])
			
			TP = cnf_av[class_current][class_current]
			FP = self.forFP(class_current, cnf_av)
			FN = self.forFN(class_current, cnf_av)
			TN = self.forTN(FP, FN, class_current, cnf_av)
			
			list_forDict.extend([None])
			list_forDict.extend(self.cf_stats(TN,TP,FP,FN))
			list_forDict.extend([None])

		#try fixing this line below more
		fScore_indexes = [(4 + 6*a) for a in range(len(class_list))]

		fScore_sum = 0

		print("This is you list_forDict")
		print(list_forDict)

		"""
		for f_index in fScore_indexes:
			fScore_sum += list_forDict[f_index]

		list_forDict.extend([(fScore_sum)/len(class_list)])
		"""

		list_fScore = []

		for f_index in fScore_indexes:
			list_fScore.extend([list_forDict[f_index]])

		list_forDict.extend([np.nanmean(list_fScore)])

		self.list_statistics.append(list_forDict)
			
	def forTN(self, FP, FN, class_current, cnf_av):
		sum_TN = 0
		for i in range(len(cnf_av)):
			for j in range(len(cnf_av)):
				sum_TN += cnf_av[i][j]
		
		sum_TN -= cnf_av[class_current][class_current]
		return sum_TN
		
	def forFP(self, class_current, cnf_av):
		sum_FP = 0
		for j in range(len(cnf_av)):
			sum_FP += cnf_av[class_current][j]
			
		sum_FP -= cnf_av[class_current][class_current]
		return sum_FP

	def forFN(self, class_current, cnf_av):
		sum_FN = 0
		for i in range(len(cnf_av)):
			sum_FN += cnf_av[i][class_current]
			
		sum_FN -= cnf_av[class_current][class_current]
		return sum_FN

	def cf_stats(self, TN, TP, FP, FN):
		"""
		args:
		---TN int for True Negative
		---TP int for True Positive
		---FP int for False Positive
		---FN int for False Negative
		does:
		---calculates statistics as a way to measure and evaluate the performance of the classifiers
		return:
		---list_return=[((TP + TN) / Total), (Precision), (Recall), (2 * ((Precision * Recall) / (Precision + Recall)))]
			---((TP + TN) / Total)) == Accuracy
			--- Precision
			--- Recall
			---(2 * ((Precision * Recall) / (Precision + Recall))) == F1 Score

		---
		"""

		"""
		AN = TN + FP
		AP = TN + FN
		PN = TN + FN
		PP = TP + FP
		"""
		Total = TN + TP + FP + FN
		Recall = (TP / (TP + FN))
		Precision = (TP / (TP + FP))

		list_return=[((TP + TN) / Total), (Precision), (Recall), (2 * ((Precision * Recall) / (Precision + Recall)))]
		return list_return

	def plot_confusion_matrix(self,cm, classes, title, htmlfolder, classifier_name, classifier_type):
		"""
		args:
		---cm is the "cnf_matrix" which is a np array of numerical values for the confusion matrix
		---classes is the class_list which is a list of the classes ie. [N,P] or [Aerobic, Anaerobic, Facultative]
		---title is a "heading" that appears on the image
		---classifier_name is the classifier name and is what the saved .png file name will be
		does:
		---creates a confusion matrix .png file and saves it
		return:
		---N/A but instead creates an .png file in tmp
		"""
		plt.rcParams.update({u'font.size': 18})
		fig = plt.figure()
		ax = fig.add_subplot(figsize=(4.5,4.5))
		#fig, ax = plt.subplots(figsize=(4.5,4.5))
		sns.set(font_scale=1.2)
		sns_plot = sns.heatmap(cm, annot=True, ax = ax, cmap=u"Blues", fmt='g'); #annot=True to annotate cells
		ax = sns_plot
		ax.set_xlabel(u'Predicted labels'); ax.set_ylabel(u'True labels');
		ax.set_title(title);
		ax.xaxis.set_ticklabels(classes); ax.yaxis.set_ticklabels(classes);
		#ax.xaxis.set_horizontalalignment('center'), ax.yaxis.set_verticalalignment('center')
		#ax.savefig(classifier_name+".png", format='png')

		fig = sns_plot.get_figure()
		#fig.savefig(u"./pics/" + classifier_name +u".png", format=u'png')
		fig.savefig(os.path.join(self.scratch, 'forHTML', htmlfolder, classifier_name +u".png"), format=u'png')

		if classifier_type == "DecisionTreeClassifier":
			fig.savefig(os.path.join(self.scratch, 'forHTML','html2folder', classifier_name +u".png"), format=u'png')

	def to_HTML_Statistics(self, class_list, classifier_name, known = "", additional = False, for_ensemble = False):
		"""
		args:
		---additional is a boolean and is used to indicate if this method is being called to make html2
		does:
		---the statistics that were calculated and stored into lists are converted into a dataframe table --> html page
		return:
		---N/A but instead creates an html file in tmp
		"""


		#self.counter = self.counter + 1

		if for_ensemble:
			statistics_dict = {}

			print("Here is the list_name")
			print(self.list_name)

			en_index = self.list_name.index(known)

			ensemble_index = [self.list_name[en_index]]
			ensembleStatistic_index = [self.list_statistics[en_index]]
			
			for i, j in izip(ensemble_index, ensembleStatistic_index):
				statistics_dict[i] = j

			data = statistics_dict


			my_index = []

			for class_current in range(len(class_list)):
				my_index.extend([class_list[class_current], u'Accuracy:', u'Precision:', u'Recall:', u'F1 score::', None])
			
			my_index.append(u'Average F1')

			df = pd.DataFrame(data, index=my_index)
			df.to_html(os.path.join(self.scratch, 'forHTML', 'html4folder', 'ensembleStatistics.html'), table_id = "ensembleStatistics", classes =["table", "table-striped", "table-bordered"])

			file = open(os.path.join(self.scratch, 'forHTML', 'html4folder', 'ensembleStatistics.html'), u'r')
			allHTML = file.read()
			file.close()

			new_allHTML = re.sub(r'NaN', r'', allHTML)

			file = open(os.path.join(self.scratch, 'forHTML', 'html4folder', 'ensembleStatistics.html'), u'w')
			file.write(new_allHTML)
			file.close

			return 0

		if not additional:

			print u"I am inside not additional"

			statistics_dict = {}

			#print(self.list_name)
			#print(self.list_statistics)

			for i, j in izip(self.list_name, self.list_statistics):
				statistics_dict[i] = j

			data = statistics_dict

			my_index = []

			for class_current in range(len(class_list)):
				my_index.extend([class_list[class_current], u'Accuracy:', u'Precision:', u'Recall:', u'F1 score::', None])
			
			my_index.append(u'Average F1')

			"""
			if class_list.__len__() == 3:
				my_index = [class_list[0], u'Accuracy:', u'Precision:', u'Recall:', u'F1 score::', None, class_list[1], u'Accuracy:',
						u'Precision:', u'Recall:', u'F1 score::', None, class_list[2], u'Accuracy:', u'Precision:', u'Recall:',
						u'F1 score::', u'Average F1']

			if class_list.__len__() == 2:
				my_index = [u'Accuracy:', u'Precision:', u'Recall:', u'F1 score::']
			"""

			df = pd.DataFrame(data, index=my_index)

			df.to_html(os.path.join(self.scratch, 'forHTML', 'html1folder','newStatistics.html'), table_id = "newStatistics", classes =["table", "table-striped", "table-bordered"])

			df['Max'] = df.idxmax(1)

			#print("Here is df[Max]")
			#print(df['Max'])

			best_classifier_str = df['Max'].iloc[-1]


			file = open(os.path.join(self.scratch, 'forHTML', 'html1folder','newStatistics.html'), u'r')
			allHTML = file.read()
			file.close()

			new_allHTML = re.sub(r'NaN', r'', allHTML)

			file = open(os.path.join(self.scratch, 'forHTML', 'html1folder','newStatistics.html'), u'w')
			file.write(new_allHTML)
			file.close

			return best_classifier_str

		if additional:
			statistics_dict = {}

			for name, name_index in izip(self.list_name, range(self.list_name.__len__())):
				if classifier_name + "_DecisionTreeClassifier" == name:
					DTClf_index = name_index
				if known == name:
					BestClf_index = name_index

			#DecisionTreeClassifier_index =

			#neededIndex = [2, 3, self.list_name.__len__() - 2, self.list_name.__len__() -1]
			#neededIndex = [self.list_name.__len__() - 2, self.list_name.__len__() -1]

			try:
				neededIndex = [DTClf_index, BestClf_index, self.list_name.__len__() - 2, self.list_name.__len__() -1]
			except:
				neededIndex = [0, self.list_name.__len__() - 2, self.list_name.__len__() -1]




			sub_list_name = [self.list_name[i] for i in neededIndex]
			sub_list_statistics = [self.list_statistics[i] for i in neededIndex]

			for i, j in izip(sub_list_name, sub_list_statistics):
				statistics_dict[i] = j

			data = statistics_dict


			my_index = []

			for class_current in range(len(class_list)):
				my_index.extend([class_list[class_current], u'Accuracy:', u'Precision:', u'Recall:', u'F1 score::', None])
			
			my_index.append(u'Average F1')

			"""
			if class_list.__len__() == 3:
				my_index = [class_list[0], u'Accuracy:', u'Precision:', u'Recall:', u'F1 score::', None, class_list[1], u'Accuracy:',
						u'Precision:', u'Recall:', u'F1 score::', None, class_list[2], u'Accuracy:', u'Precision:', u'Recall:',
						u'F1 score::', u'Average F1']

			if class_list.__len__() == 2:
				my_index = [u'Accuracy:', u'Precision:', u'Recall:', u'F1 score::']
			"""

			df = pd.DataFrame(data, index=my_index)
			df.to_html(os.path.join(self.scratch, 'forHTML', 'html2folder', 'postStatistics.html'), table_id = "postStatistics", classes =["table", "table-striped", "table-bordered"])

			df['Max'] = df.idxmax(1)
			best_classifier_str = df['Max'].iloc[-1]

			file = open(os.path.join(self.scratch, 'forHTML', 'html2folder', 'postStatistics.html'), u'r')
			allHTML = file.read()
			file.close()

			new_allHTML = re.sub(r'NaN', r'', allHTML)

			file = open(os.path.join(self.scratch, 'forHTML', 'html2folder', 'postStatistics.html'), u'w')
			file.write(new_allHTML)
			file.close

			return best_classifier_str

	#### Below is code for tuning Decision Tree ####

	def tune_Decision_Tree(self,classifierTest_paramsInput, best_classifier_str = None):
		"""
		args:
		---classifierTest_paramsInput is same dictionary as classifierTest_params passed in with few updates
		---best_classifier_str (optional) passed to to_HTML_Statistics in case the best classifier was known so a column for it could be created in the table
		does:
		---by looping through various parameters (1. depth 2. criterion) it selects best configuration
		---calls tree_code
		return:
		---N/A but main function is just to figure out "workhorse"
		"""
		
		classifierTest_params = {
				'ctx' : classifierTest_paramsInput['ctx'],
				'current_ws' : classifierTest_paramsInput['current_ws'],
				#'classifier' : ,
				'classifier_type' : classifierTest_paramsInput['classifier_type'],
				'classifier_name' : classifierTest_paramsInput['classifier_name'],
				'my_mapping' : classifierTest_paramsInput['my_mapping'],
				'master_Role' : classifierTest_paramsInput['master_Role'],
				'splits' : classifierTest_paramsInput['splits'],
				'train_index' : classifierTest_paramsInput['train_index'],
				'test_index' : classifierTest_paramsInput['test_index'],
				'all_attributes' : classifierTest_paramsInput['all_attributes'],
				'all_classifications' : classifierTest_paramsInput['all_classifications'],
				'class_list' : classifierTest_paramsInput['class_list'],
				'htmlfolder' : classifierTest_paramsInput['htmlfolder'],
				'print_cfm' : True,
				'training_set_ref' : classifierTest_paramsInput['training_set_ref'],
				'description' : classifierTest_paramsInput['description']
				}

		#below code is for gini-criterion

		classifierTest_params['classifier_name'] = classifierTest_paramsInput['classifier_name'] + u"DecisionTreeClassifier"
		classifierTest_params['print_cfm'] = False

		val = np.zeros(12)
		test_av = np.zeros(12)
		test_std = np.zeros(12)
		val_av = np.zeros(12)
		val_std = np.zeros(12)
		for d in xrange(1, 12):
			classifierTest_params['classifier'] = DecisionTreeClassifier(random_state=0, max_depth=d)
			val[d] = d
			(test_av[d], test_std[d], val_av[d], val_std[d]) = self.classifierTest(classifierTest_params)

		fig, ax = plt.subplots(figsize=(6, 6))
		plt.errorbar(val[1:], test_av[1:], yerr=test_std[1:], fmt=u'o', label=u'Training set')
		plt.errorbar(val[1:], val_av[1:], yerr=val_std[1:], fmt=u'o', label=u'Testing set')
		ax.set_ylim(ymin=0.0, ymax=1.1)
		ax.set_title(u"Gini Criterion")
		plt.xlabel(u'Tree depth', fontsize=12)
		plt.ylabel(u'Accuracy', fontsize=12)
		plt.legend(loc=u'lower left')
		#plt.savefig(u"./pics/"+ global_target +u"_gini_depth-met.png")
		#fig.savefig(u"/kb/module/work/tmp/pics/" + classifier_name +u".png", format=u'png')
		plt.savefig(os.path.join(self.scratch, 'forHTML', 'html2folder', classifierTest_paramsInput['classifier_name'] +u"_gini_depth-met.png"))

		gini_best_index = np.argmax(val_av)
		print gini_best_index
		gini_best = np.amax(val_av)

		#below code is for entropy-criterion

		classifierTest_params['classifier_name'] = classifierTest_paramsInput['classifier_name'] + u"DecisionTreeClassifier"
		classifierTest_params['print_cfm'] = False

		val = np.zeros(12)
		test_av = np.zeros(12)
		test_std = np.zeros(12)
		val_av = np.zeros(12)
		val_std = np.zeros(12)
		for d in xrange(1, 12):
			classifierTest_params['classifier'] = DecisionTreeClassifier(random_state=0, max_depth=d, criterion=u'entropy')
			val[d] = d
			(test_av[d], test_std[d], val_av[d], val_std[d]) = self.classifierTest(classifierTest_params)

		fig, ax = plt.subplots(figsize=(6, 6))
		plt.errorbar(val[1:], test_av[1:], yerr=test_std[1:], fmt=u'o', label=u'Training set')
		plt.errorbar(val[1:], val_av[1:], yerr=val_std[1:], fmt=u'o', label=u'Testing set')
		ax.set_ylim(ymin=0.0, ymax=1.1)
		ax.set_title(u"Entropy Criterion")
		plt.xlabel(u'Tree depth', fontsize=12)
		plt.ylabel(u'Accuracy', fontsize=12)
		plt.legend(loc=u'lower left')
		#plt.savefig(u"./pics/"+ global_target +u"_entropy_depth-met.png")
		plt.savefig(os.path.join(self.scratch, 'forHTML', 'html2folder', classifierTest_paramsInput['classifier_name'] +u"_entropy_depth-met.png"))

		entropy_best_index = np.argmax(val_av)
		print entropy_best_index
		entropy_best = np.amax(val_av)


		#gini_best_index = 4
		#entropy_best_index = 3

		classifierTest_params['classifier'] = DecisionTreeClassifier(random_state=0, max_depth=gini_best_index, criterion=u'gini')
		classifierTest_params['classifier_name'] = classifierTest_paramsInput['classifier_name'] + u"_DecisionTreeClassifier_gini"
		classifierTest_params['print_cfm'] = True
		self.classifierTest(classifierTest_params)
		

		classifierTest_params['classifier'] = DecisionTreeClassifier(random_state=0, max_depth=entropy_best_index, criterion=u'entropy')
		classifierTest_params['classifier_name'] = classifierTest_paramsInput['classifier_name'] + u"_DecisionTreeClassifier_entropy"
		classifierTest_params['print_cfm'] = True
		self.classifierTest(classifierTest_params)

		self.to_HTML_Statistics(classifierTest_params['class_list'], classifierTest_paramsInput['classifier_name'], known = best_classifier_str,additional=True)

		if gini_best > entropy_best:
			self.tree_code(DecisionTreeClassifier(random_state=0, max_depth=gini_best_index, criterion=u'gini'), classifierTest_params['all_attributes'], classifierTest_params['all_classifications'], classifierTest_params['master_Role'], classifierTest_params['class_list'])
		else:
			self.tree_code(DecisionTreeClassifier(random_state=0, max_depth=entropy_best_index, criterion=u'entropy'), classifierTest_params['all_attributes'], classifierTest_params['all_classifications'], classifierTest_params['master_Role'], classifierTest_params['class_list'])


	def tree_code(self, optimized_tree, all_attributes, all_classifications, master_Role, class_list, spacer_base=u"    "):
		"""
		args:
		---optimized_tree this is a DecisionTree object that has been tuned
		---spacer_base is string physically acting as a spacer
		does:
		---Produce psuedo-code for decision tree - based on http://stackoverflow.com/a/30104792.
		---calls printTree
		return:
		---N/A but prints out a visual of what the DecisionTree object looks like on the inside
		"""

		tree = optimized_tree 
		#tree = DecisionTreeClassifier(random_state=0, max_depth=3, criterion='entropy')

		tree.fit(all_attributes, all_classifications)
		feature_names = master_Role
		target_names = class_list

		left = tree.tree_.children_left
		right = tree.tree_.children_right
		threshold = tree.tree_.threshold
		features = [feature_names[i] for i in tree.tree_.feature]
		value = tree.tree_.value

		def recurse(left, right, threshold, features, node, depth):
			spacer = spacer_base * depth
			if (threshold[node] != -2):
				print spacer + u"if ( " + features[node] + u" <= " + \
					  unicode(threshold[node]) + u" ) {"
				if left[node] != -1:
					recurse(left, right, threshold, features,
								 left[node], depth + 1)
				print spacer + u"}\n" + spacer + u"else {"
				if right[node] != -1:
					recurse(left, right, threshold, features,
								 right[node], depth + 1)
				print spacer + u"}"
			else:
				target = value[node]
				for i, v in izip(np.nonzero(target)[1],
								target[np.nonzero(target)]):
					target_name = target_names[i]
					target_count = int(v)
					print spacer + u"return " + unicode(target_name) + \
						  u" ( " + unicode(target_count) + u" examples )"

		recurse(left, right, threshold, features, 0, 0)

		self.printTree(tree, u"NAMEmyTreeLATER", master_Role, class_list)

	def printTree(self,tree, pass_name, master_Role, class_list):
		"""
		args:
		---tree is a DecisionTree object that has already been tuned
		---pass_name is a string for what you want the tree named as (but this is not where the creation happens just pass)
		---master_Role (same as classifierTest)
		---class_list (same as classifierTest)
		does:
		---using graphviz feature it is able to geneate the dot file that has an "ugly" version of the tree inside
		---call the parse_lookNice
		return:
		---N/A just makes an "ugly" dot file.
		"""

		not_dotfile = StringIO.StringIO()
		export_graphviz(tree, out_file=not_dotfile, feature_names=master_Role,
						class_names=class_list)

		contents = not_dotfile.getvalue()
		not_dotfile.close()

		dotfile = open(os.path.join(self.scratch, 'dotFolder', 'mydotTree.dot'), u'w')
		dotfile.write(contents)
		dotfile.close()

		self.parse_lookNice(pass_name,tree, master_Role, class_list)

	def parse_lookNice(self, name, tree, master_Role, class_list):
		"""
		args:
		---name is a string that is what you want the DecisionTree image saved as
		---tree is a DecisionTree object that has already been tuned
		---master_Role (same as classifierTest)
		---class_list (same as classifierTest)
		does:
		---this cleans up the dot file to produce a more visually appealing tree figure using graphviz
		return:
		---N/A but saves a .png of the name in the tmp folder
		"""
		f = open(os.path.join(self.scratch, 'dotFolder', 'mydotTree.dot'), u"r")
		allStr = f.read()
		f.close()
		new_allStr = allStr.replace(u'\\n', u'')

		first_fix = re.sub(ur'(\w\s\[label="[\w\s.,:\'\/()-]+)<=([\w\s.\[\]=,]+)("] ;)',
						   ur'\1 (Absent)" , color="0.650 0.200 1.000"] ;', new_allStr)
		second_fix = re.sub(ur'(\w\s\[label=")(.+?class\s=\s)', ur'\1', first_fix)

		# nominal fixes like color and shape
		third_fix = re.sub(ur'shape=box] ;', ur'shape=Mrecord] ; node [style=filled];', second_fix)

		color_set = []
		for class_current in range(len(class_list)):
			color_set.extend(['%.4f'%random.uniform(0, 1) + " " + '%.4f'%random.uniform(0, 1) + " " + '0.900'])

		for class_current, my_color in izip(range(len(class_list)),color_set):
			third_fix = re.sub(ur'(\w\s\[label="%s")' % class_list[class_current], ur'\1, color = "%s"' % my_color, third_fix)

		f = open(os.path.join(self.scratch, 'dotFolder', 'niceTree.dot'), u"w")
		f.write(third_fix)
		f.close()

		os.system(u'dot -Tpng ' + os.path.join(self.scratch, 'dotFolder', 'niceTree.dot') + ' >  '+ os.path.join(self.scratch, 'forHTML', 'html2folder', name + u'.png '))
		self.top20Important(tree, master_Role)

		"""
		if class_list.__len__() == 3:
			fourth_fix = re.sub(ur'(\w\s\[label="%s")' % class_list[0], ur'\1, color = "0.5176 0.2314 0.9020"', third_fix)
			fifth_fix = re.sub(ur'(\w\s\[label="%s")' % class_list[1], ur'\1, color = "0.5725 0.6118 1.0000"', fourth_fix)
			sixth_fix = re.sub(ur'(\w\s\[label="%s")' % class_list[2], ur'\1, color = "0.5804 0.8824 0.8039"', fifth_fix)
			f = open(os.path.join(self.scratch, 'dotFolder', 'niceTree.dot'), u"w")
			f.write(sixth_fix)
			f.close()

			os.system(u'dot -Tpng ' + os.path.join(self.scratch, 'dotFolder', 'niceTree.dot') + ' >  '+ os.path.join(self.scratch, 'forHTML', 'html2folder', name + u'.png '))
			self.top20Important(tree, master_Role)

		if class_list.__len__() == 2:
			fourth_fix = re.sub(ur'(\w\s\[label="%s")' % class_list[0], ur'\1, color = "0.5176 0.2314 0.9020"', third_fix)
			fifth_fix = re.sub(ur'(\w\s\[label="%s")' % class_list[1], ur'\1, color = "0.5725 0.6118 1.0000"', fourth_fix)
			f = open(os.path.join(self.scratch, 'dotFolder', 'niceTree.dot'), u"w")
			f.write(fifth_fix)
			f.close()

			os.system(u'dot -Tpng ' + os.path.join(self.scratch, 'dotFolder', 'niceTree.dot') + ' >  '+ os.path.join(self.scratch, 'forHTML', 'html2folder', name + u'.png '))
			self.top20Important(tree, master_Role)
		"""



	def top20Important(self,tree, master_Role):
		"""
		args:
		---tree is a DecisionTree object that has already been tuned
		---master_Role (same as classifierTest)
		does:
		---find the list of the top20 most important roles in determining classification
		return:
		---creates a dataframe that is displayed in the html report
		"""

		data = {'attribute_list': master_Role, 'importance': tree.feature_importances_}

		forImportance = pd.DataFrame.from_dict(data)

		#display the top 20 most important functional roles and weights
		top20 = forImportance.sort_values('importance', ascending=False)['attribute_list'].head(20)
		top20_weight = forImportance.sort_values('importance', ascending=False)['importance'].head(20)*100
		list_top20 = list(top20)
		list_top20_weight = list(top20_weight)

		df_top20 = pd.DataFrame.from_dict({'Top 20 Prioritized Roles': list_top20, 'Weights': list_top20_weight})

		old_width = pd.get_option('display.max_colwidth')
		pd.set_option('display.max_colwidth', -1)
		df_top20.to_html(os.path.join(self.scratch, 'forHTML', 'html2folder', 'top20.html'), index=False, justify='center', table_id = "top20", classes =["table", "table-striped", "table-bordered"])
		pd.set_option('display.max_colwidth', old_width)

		#create a downloadable link to all functional roles and weights
		top = forImportance.sort_values('importance', ascending=False)['attribute_list']
		top_weight = forImportance.sort_values('importance', ascending=False)['importance']
		list_top = list(top)
		list_top_weight = list(top_weight)

		df_top = pd.DataFrame.from_dict({'Top Prioritized Roles': list_top, 'Weights': list_top_weight})

		writer = pd.ExcelWriter(os.path.join(self.scratch, 'forHTML', 'forDATA', 'prioritized_weights.xlsx'), engine='xlsxwriter')
		df_top.to_excel(writer, sheet_name='Sheet1')
		writer.save()

	### Extra methods being used 
	def _make_dir(self):
		dir_path = os.path.join(self.scratch, str(uuid.uuid4()))
		os.mkdir(dir_path)

		return dir_path

	def _download_shock(self, shock_id = None, handle_id = None):
		"""
		does:
		---using kbase dfu tool to allow users to insert excel files 
		"""
		dir_path = self._make_dir()

		if(handle_id):
			file_path = self.dfu.shock_to_file({'handle_id': handle_id,
											'file_path': dir_path})['file_path']
		
		else:
			file_path = self.dfu.shock_to_file({'shock_id': shock_id,
											'file_path': dir_path})['file_path']

		return file_path

	def _upload_to_shock(self, file_path):
		"""
		does:
		---using kbase dfu tool to allow users to insert excel files 
		"""
		# dir_path = self._make_dir()

		"""print('here is the file_path')
								print(type(file_path))
								print(file_path)"""

		f2shock_out = self.dfu.file_to_shock({'file_path': file_path,
											  'make_handle': True})

		shock_id = f2shock_out.get('shock_id')
		handle_id = f2shock_out.get('handle').get('hid')

		return shock_id, handle_id

	#### HTML templates below ####

	### For Build_Classifier App

	def html_report_0(self, missingGenomes, phenotype):
		file = open(os.path.join(self.scratch, 'forZeroHTML', 'html0.html'), u"w")

		html_string = u"""
		<!DOCTYPE html>
		<html>

		<head>
		
		<link href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/4.1.3/css/bootstrap.css" rel="stylesheet">

		<style>
		table, th, td , tr{
			text-align: center;
		}
		.table tbody tr td{background:#fff /*set your own color*/} 
    	.table tbody tr th{background:#fff /*set your own color*/} 
    	.table thead tr th{background:#fff /*set your own color*/} 
		
		.container {
			max-width: 90%;
			padding-top: 50px;
      		padding-bottom: 50px;
		}
		</style>

		</head>
		<body>
		<div class="container">

		<h1 style="text-align:center;"> Missing Genomes in Training Data </h1>

		<p> Missing genomes are listed below (ie. these were included in your excel / pasted file but are not present in the workspace).
		A training set object was created regardless of if there were missing genomes. In the event that there were missing genomes they were excluded  </p>
		<p> The missing genomes are: """ + str(missingGenomes) + """ </p>
		
		<br>

		<p>Below is a detailed table which shows Genome ID, whether it was loaded into the Narrative, its """ + phenotype + """ Classification, and if it was Added to the Training Set</p>

		"""
		file.write(html_string)

		another_file = open(os.path.join(self.scratch, 'forZeroHTML','four_columns.html'), u"r")
		all_str = another_file.read()
		another_file.close()
		file.write(all_str)

		next_str =u"""
		</div>
		</body>

		<script type="text/javascript" src="https://code.jquery.com/jquery-3.3.1.js"></script>
		<script type="text/javascript" src="https://cdn.datatables.net/v/bs4-4.1.1/jq-3.3.1/dt-1.10.18/b-1.5.4/b-colvis-1.5.4/b-html5-1.5.4/b-print-1.5.4/datatables.min.js"></script>
    
		<script type="text/javascript" language="javascript" class="init">
   		$(document).ready(function() {
			$('#four_columns').DataTable();
		});
		</script>
		</html>
		"""
		file.write(next_str)
		file.close()

		#return "html0.html"

	def html_report_1(self, global_target, classifier_type, classifier_name, phenotype, num_classes, best_classifier_str = None):
		"""
		does: creates an .html file that makes the frist report (first app).
		"""
		file = open(os.path.join(self.scratch, 'forHTML', 'html1folder', 'html1.html'), u"w")

		html_string = u"""
		<!DOCTYPE html>
		<html>

		<head>
		
		<link href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/4.1.3/css/bootstrap.css" rel="stylesheet">

		<style>
		.dataTables_filter {
      		width: 50%;
      		float: right;
      		text-align: right;
    	}
		table, th, td , tr{
			text-align: center;
		}
		.table tbody tr td{background:#fff /*set your own color*/} 
    	.table tbody tr th{background:#fff /*set your own color*/} 
    	.table thead tr th{background:#fff /*set your own color*/} 

		.container {
			max-width: 90%;
			padding-top: 50px;
      		padding-bottom: 50px;
		}
		.row {
			display: flex;
			flex-wrap: wrap;
			padding: 0 4px;
		}

		/* Create two equal columns that sits next to each other */
		.column {
			flex: 50%;
			padding: 0 4px;
		}

		.column img {
			margin-top: 8px;
			vertical-align: middle;
		}
		</style>

		</head>
		<body>
		<div class="container">

		<h1 style="text-align:center;">""" + global_target + """ - Classifier</h1> """

		file.write(html_string)

		if classifier_type == u"run_all":
			
			next_str = u"""
			<p style="text-align:center; font-size:160%;"> """ + global_target + """ Classification models were created based on the selected trainging set object, classification algorithim, and attribute.
			The effectiveness of each model to predict """+ phenotype +""" is displayed by a confusion matrix* and relevant statistical measures below. </p>

			<p> The selected classifiers were: </p>
			<ul>
				<li>K-Nearest-Neighbors Classifier</li>
				<li>Logistic Regression Classifier</li>
				<li>Naive Gaussian Bayes Classifier</li>
				<li>Support Vector Machine (SVM) Classifier</li>
				<li>Decision Tree Classifier</li>
				<li>Neural Network </li>
			</ul> 

			<p> Further more, advanced options were not selected so no feature selection and parameter optimization was conducted. </p>
			"""

			file.write(next_str)

		else :

			next_str = u"""
			<p style="text-align:center; font-size:160%;"> """ + global_target + """ Classification models were created based on the selected trainging set object, classification algorithim, and attribute.
			The effectiveness of each model is displayed by a confusion matrix* and relevant statistical measures below. </p>

			<p> The selected classifiers were: </p>
			<ul>
				<li>""" + classifier_type + """</li>
			</ul> 

			<p> Further more, advanced options were not selected so no feature selection and parameter optimization was conducted. </p>
			"""

			file.write(next_str)

		next_str = u"""
		<p style="font-size:110%;"> 
		*A confusion matrix is a table that is used to describe the performance of a classifier
		on a set of test data for which the true values are known - showing the comparision between the predicted labels and true labels. (In our case we used 
		K-fold Cross Validation and the below confusion matrices represent the "average" of k-folds.) 
		The number in each cell of the confusion matrix is the percentage of samples with a true label being classified with the predicted label.
		A strong classifier is one that has a central diagonal with the highest percentages, meaning that the majority of the predicted labels match the true label.
		</p>

		<br/>

		<p> Note that each classfication model can be downloaded by either clicking the Download button or selecting the desired model under links. As of now,
		the format of these models are python pickel object created from the sklearn library</p>
		"""

		file.write(next_str)

		if classifier_type == u"run_all":
			if num_classes >= 6:
				next_str = u"""
				<p style="color: #ff5050;> Sorry, we cannot display confusion matricies for classifiers with greater than 6 classes. However statistics are still produced below. </p>
				"""
				file.write(next_str)

			else:
				next_str = u"""
				<div class="row"> 
					<div class="column">
						<p style="text-align:left; font-size:160%;">A.) K-Nearest-Neighbors Classifier <a href="../forDATA/""" + classifier_name + """_KNeighborsClassifier.pickle" download>  (Download) </a> </p>
						<img src=" """+ classifier_name +"""_KNeighborsClassifier.png"  alt="Snow" style="width:100%">
						<p style="text-align:left; font-size:160%;">C.) Naive Gaussian Bayes Classifier <a href="../forDATA/""" + classifier_name + """_GaussianNB.pickle" download> (Download) </a> </p>
						<img src=" """+ classifier_name +"""_GaussianNB.png" alt="Snow" style="width:100%">
						<p style="text-align:left; font-size:160%;">E.) Decision Tree Classifier <a href="../forDATA/""" + classifier_name + """_DecisionTreeClassifier.pickle" download> (Download) </a> </p>
						<img src=" """+ classifier_name +"""_DecisionTreeClassifier.png" alt="Snow" style="width:100%">
					</div>

					<div class="column">
						<p style="text-align:left; font-size:160%;">B.) Logistic Regression Classifier <a href="../forDATA/""" + classifier_name + """_LogisticRegression.pickle" download> (Download) </a> </p>
						<img src=" """+ classifier_name +"""_LogisticRegression.png" alt="Snow" style="width:100%">
						<p style="text-align:left; font-size:160%;">D.) Support Vector Machine (SVM) Classifier <a href="../forDATA/""" + classifier_name + """_SVM.pickle" download> (Download) </a> </p>
						<img src=" """+ classifier_name +"""_SVM.png" alt="Snow" style="width:100%">
						<p style="text-align:left; font-size:160%;">F.) Neural Network Classifier <a href="../forDATA/""" + classifier_name + """_NeuralNetwork.pickle" download> (Download) </a> </p>
						<img src=" """+ classifier_name +"""_NeuralNetwork.png" alt="Snow" style="width:100%">
					</div> 
				</div>

				"""
				file.write(next_str)

			next_str = u"""
			<br>

			<p style="font-size:160%;">Below are statistics for each class based on each model: in the form of Accuracy, Precision, Recall and F1 Score. The statistics were derived from the confusion matricies.</p>
			<p style="font-size:100%;">Defintion of key statistics:</p> 
			
			<ul>
				<li>Accuracy - How often is the classifier correct</li>
				<li>Precision - When predition is positive how often is it correct</li>
				<li>Recall - When the condition is correct how often is it correct</li>
				<li>F1 Score - The is a weighted average of recall and precision</li>
			</ul>         
			"""
			file.write(next_str)

			another_file = open(os.path.join(self.scratch, 'forHTML', 'html1folder', 'newStatistics.html'), u"r")
			all_str = another_file.read()
			another_file.close()

			file.write(all_str)

			next_str = u"""
			<br>

			<p style="text-align:center; font-size:100%;">  The best model (based on the highest average F1 score) is: """ + unicode(best_classifier_str) + """ </p>
			"""

			file.write(next_str)

		else:
			if num_classes >= 9:
				next_str = u"""
				<p style="color: #ff5050;"> Sorry, we cannot display confusion matricies for classifiers with greater than 9 classes. However statistics are still produced below. </p>
				"""
				file.write(next_str)
			else:
				next_str = u"""
				<div class="row">
					<div class="column">
						<p style="text-align:left; font-size:160%;">A.) """ + classifier_type + """ <a href="../forDATA/""" + classifier_name + """.pickle" download> (Download) </a> </p>
						<img src=" """+ classifier_name +""".png" alt="Snow" style="width:100%">
					</div>
					<div class="column">
				"""
				file.write(next_str)

			next_str = u"""

			<br>

			<p style="font-size:160%;">Below are statistics for each class, for each model, in the form of Accuracy, Precision, Recall and F1 Score. The statistics were derived from the confusion matricies.</p>
			<p style="font-size:100%;">Defintion of key statistics:</p> 
			
			<ul>
				<li>Accuracy - How often is the classifier correct</li>
				<li>Precision - When predition is positive how often is it correct</li>
				<li>Recall - When the condition is correct how often is it correct</li>
				<li>F1 Score - The is a weighted average of recall and precision</li>
			</ul>           
			"""
			file.write(next_str)

			another_file = open(os.path.join(self.scratch, 'forHTML', 'html1folder', 'newStatistics.html'), u"r")
			all_str = another_file.read()
			another_file.close()

			file.write(all_str)

			next_str = u"""
			</div>
			</div>
			"""
			file.write(next_str)

		next_str = u"""
		</div>
		</body>

		<script type="text/javascript" src="https://code.jquery.com/jquery-3.3.1.js"></script>
		<script type="text/javascript" src="https://cdn.datatables.net/v/bs4-4.1.1/jq-3.3.1/dt-1.10.18/b-1.5.4/b-colvis-1.5.4/b-html5-1.5.4/b-print-1.5.4/datatables.min.js"></script>
		<script type="text/javascript" src="https://cdn.datatables.net/fixedcolumns/3.2.6/js/dataTables.fixedColumns.min.js"></script>

		<script type="text/javascript" language="javascript" class="init">
   		$(document).ready(function() {
			$('#newStatistics').DataTable({
        		ordering:		false,
				scrollY:        true,
				scrollX:        true,
				scrollCollapse: true,
				paging:         false,
				fixedColumns:   true
			});
		});
		</script>
		</html>
		"""

		file.write(next_str)

		file.close()

	def html_report_2(self, global_target, classifier_name, num_classes, best_classifier_str = None):
		"""
		does: creates an .html file that makes the second report (first app).
		"""
		file = open(os.path.join(self.scratch, 'forHTML', 'html2folder', 'html2.html'), u"w")

		html_string = u"""
		<!DOCTYPE html>
		<html>

		<head>
		
		<link href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/4.1.3/css/bootstrap.css" rel="stylesheet">

		<style>
		.dataTables_filter {
      		width: 50%;
      		float: right;
      		text-align: right;
    	}
		table, th, td , tr{
			text-align: center;
		}
		.table tbody tr td{background:#fff /*set your own color*/} 
    	.table tbody tr th{background:#fff /*set your own color*/} 
    	.table thead tr th{background:#fff /*set your own color*/} 

		.container {
			max-width: 90%;
			padding-top: 50px;
      		padding-bottom: 50px;
		}
		.row {
			display: flex;
			flex-wrap: wrap;
			padding: 0 4px;
		}

		/* Create two equal columns that sits next to each other */
		.column {
			flex: 50%;
			padding: 0 4px;
		}

		.column img {
			margin-top: 8px;
			vertical-align: middle;
		}		
		</style>

		</head>
		<body>
		<div class="container">

		<h1 style="text-align:center;">""" + global_target + """ - Decision Tree Tuning</h1>

		<!-- <h2>Maybe we can add some more text here later?</h2> -->
		<!--<p>How to create side-by-side images with the CSS float property:</p> -->

		<p style="text-align:left; font-size:160%;">  We tune the Decision Tree based on two hyperparameters: Tree Depth and Criterion (quality of a split) </p>
		<p style="text-align:left; font-size:100%;">  The two criterion were "gini" which uses the Gini impurity socre and "entropy" which uses information gain score. </p>
		"""

		file.write(html_string)

		next_str = u"""

		<div class="row">
		  <div class="column">
			  <p style="text-align:left; font-size:160%;">Training vs Testing Score on Gini Criterion </p>
			<img src=" """+ classifier_name +"""_gini_depth-met.png" alt="Snow" style="width:100%">
		  </div>
		  <div class="column">
			  <p style="text-align:left; font-size:160%;">Training vs Testing Score on Entropy Criterion</p>
			<img src=" """+ classifier_name +"""_entropy_depth-met.png" alt="Snow" style="width:100%">
		  </div>
		</div>
		"""

		file.write(next_str)

		if (best_classifier_str == None) or ("DecisionTreeClassifier" in best_classifier_str):
			if num_classes >= 6:
				next_str = u"""
				<p style="color: #ff5050;"> Sorry, we cannot display confusion matricies for classifiers with greater than 6 classes. However statistics are still produced below. </p>
				"""
				file.write(next_str)

			else:	
				next_str = u"""
				<p style="text-align:center; font-size:160%;">  The effectiveness of each model is displayed by a confusion matrix. We also include the 
				original Decision Tree Classifier model as a baseline. </p>

				<div class="row">
				<div class="column">
					<p style="text-align:left; font-size:160%;">A.) Decision Tree Classifier <a href="../forDATA/""" + classifier_name + """_DecisionTreeClassifier.pickle" download> (Download) </a> </p>
					<img src=" """+ classifier_name +"""_DecisionTreeClassifier.png" alt="Snow" style="width:100%">
				</div>

				</div>

				<div class="row">
					<div class="column">
						<p style="text-align:left; font-size:160%;">B.) Decision Tree Classifier - Gini <a href="../forDATA/""" + classifier_name + """_DecisionTreeClassifier_gini.pickle" download> (Download) </a> </p>
						<img src=" """+ classifier_name +"""_DecisionTreeClassifier_gini.png" alt="Snow" style="width:100%">
					</div>
					<div class="column">
						<p style="text-align:left; font-size:160%;">C.) Decision Tree Classifier - Entropy <a href="../forDATA/""" + classifier_name + """_DecisionTreeClassifier_entropy.pickle" download> (Download) </a> </p>
						<img src=" """+ classifier_name +"""_DecisionTreeClassifier_entropy.png" alt="Snow" style="width:100%">
					</div>
				</div>
				"""
				file.write(next_str)
		else:
			if num_classes >= 6:
				next_str = u"""
				<p style="color: #ff5050;"> Sorry, we cannot display confusion matricies for classifiers with greater than 6 classes. However statistics are still produced below. </p>
				"""
				file.write(next_str)

			else:	
				next_str = u"""<p style="text-align:center; font-size:160%;">  The effectiveness of each model is displayed by a confusion matrix. We also include the 
				original Decision Tree Classifier model as a baseline  and """+ best_classifier_str + """ as it showed the highest average F1 Score </p>

				<div class="row">
				<div class="column">
					<p style="text-align:left; font-size:160%;">A.) Decision Tree Classifier <a href="../forDATA/""" + classifier_name + """_DecisionTreeClassifier.pickle" download> (Download) </a> </p>
					<img src=" """+ classifier_name +"""_DecisionTreeClassifier.png" alt="Snow" style="width:100%">
				</div>

				<div class="column">
					<p style="text-align:left; font-size:160%;">B.) """+ best_classifier_str + """ <a href="../forDATA/""" + best_classifier_str + """.pickle" download> (Download) </a> </p>
					<img src=" """+ best_classifier_str + """.png" alt="Snow" style="width:100%">
					</div>
				</div>

				<div class="row">
					<div class="column">
						<p style="text-align:left; font-size:160%;">C.) Decision Tree Classifier - Gini <a href="../forDATA/""" + classifier_name + """_DecisionTreeClassifier_gini.pickle" download> (Download) </a> </p>
						<img src=" """+ classifier_name +"""_DecisionTreeClassifier_gini.png" alt="Snow" style="width:100%">
					</div>
					<div class="column">
						<p style="text-align:left; font-size:160%;">D.) Decision Tree Classifier - Entropy <a href="../forDATA/""" + classifier_name + """_DecisionTreeClassifier_entropy.pickle" download> (Download) </a> </p>
						<img src=" """+ classifier_name +"""_DecisionTreeClassifier_entropy.png" alt="Snow" style="width:100%">
					</div>
				</div>
				"""
				file.write(next_str)

		next_str= u"""
		<br>
		<p style="font-size:160%;">Below are statistics for each class, for each model, in the form of Accuracy, Precision, Recall and F1 Score. The statistics were derived from the confusion matricies.</p>
		"""
		file.write(next_str)

		another_file = open(os.path.join(self.scratch, 'forHTML', 'html2folder', 'postStatistics.html'), u"r")
		all_str = another_file.read()
		another_file.close()
		file.write(all_str)

		next_str= u"""
		<br>
		<p style="font-size:160%;"> Below is a visual for the Decision Tree Classifier with the highest F1 Score.</p>
		<img src="NAMEmyTreeLATER.png" alt="Snow" style="width:100%">

		<br>
		<br>

		<p style="font-size:100%;"> Below is a list of the Top 20 Roles that were given the highest importance by the Decision Tree Classifier.</p>
		<p style="text-align:left; font-size:160%;">D.) Table with prioritized_weights <a href="../forDATA/prioritized_weights.xlsx" download> (Download) </a> </p>
		"""
		file.write(next_str)

		another_file = open(os.path.join(self.scratch, 'forHTML', 'html2folder', 'top20.html'), u"r")
		all_str = another_file.read()
		another_file.close()
		file.write(all_str)

		next_str = u"""
		</div>
		</body>

		<script type="text/javascript" src="https://code.jquery.com/jquery-3.3.1.js"></script>
		<script type="text/javascript" src="https://cdn.datatables.net/v/bs4-4.1.1/jq-3.3.1/dt-1.10.18/b-1.5.4/b-colvis-1.5.4/b-html5-1.5.4/b-print-1.5.4/datatables.min.js"></script>
    
		<script type="text/javascript" language="javascript" class="init">
		$(document).ready(function() {
			$('#postStatistics').DataTable({
        		ordering:		false,
				scrollY:        true,
				scrollX:        true,
				scrollCollapse: true,
				paging:         false,
				fixedColumns:   true
			});
			$('#top20').DataTable();
		});
		</script>
		</html>
		"""
		file.write(next_str)

		file.close()

	def html_report_4(self, global_target, classifier_name, estimators_inHTML, num_classes):
		file = open(os.path.join(self.scratch, 'forHTML', 'html4folder', 'html4.html'), u"w")
		
		html_string = u"""
		<!DOCTYPE html>
		<html>

		<head>
		
		<link href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/4.1.3/css/bootstrap.css" rel="stylesheet">

		<style>
		.dataTables_filter {
      		width: 50%;
      		float: right;
      		text-align: right;
    	}
		table, th, td , tr{
			text-align: center;
		}

		.container {
      		max-width: 90%;
			padding-top: 50px;
      		padding-bottom: 50px;
    	}
		.table tbody tr td{background:#fff /*set your own color*/} 
    	.table tbody tr th{background:#fff /*set your own color*/} 
    	.table thead tr th{background:#fff /*set your own color*/} 
		</style>

		</head>
		<body>
		<div class="container">

		<h1 style="text-align:center;">""" + global_target + """ - Ensemble Model</h1>

		<p> Below is an Ensemble Classfier based on a "hard" majority rule voting
		mechanism as its weights from the selected models:  """+ estimators_inHTML + """.  </p>

		"""

		file.write(html_string)
		
		if num_classes >= 6:
				next_str = u"""
				<p style="color: #ff5050;"> Sorry, we cannot display confusion matricies for classifiers with greater than 6 classes. However statistics are still produced below. </p>
				"""
				file.write(next_str)

		else:	
			next_str = u"""
			<div class="row">
				<div class="column">
					<p style="text-align:left; font-size:160%;"> Ensemble Classifier <a href="../forDATA/""" + classifier_name + """_Ensemble_Model.pickle" download> (Download) </a> </p>
					<img src=" """+ classifier_name +"""_Ensemble_Model.png" alt="Snow" style="width:100%">
				</div>
				<div class="column">
				</div>
			</div>
			"""
			file.write(next_str)


		another_file = open(os.path.join(self.scratch, 'forHTML', 'html4folder', 'ensembleStatistics.html'), u"r")
		all_str = another_file.read()
		another_file.close()

		file.write(all_str)


		next_str = u"""
		
		<p> The effectiveness of each model is displayed by a confusion matrix above. Furthermore statistics for each class, for each model,
		 in the form of Accuracy, Precision, Recall and F1 Score is below. The statistics were derived from the confusion matricies. </p>  
		
		</div>
		</body>

		<script type="text/javascript" src="https://code.jquery.com/jquery-3.3.1.js"></script>
		<script type="text/javascript" src="https://cdn.datatables.net/v/bs4-4.1.1/jq-3.3.1/dt-1.10.18/b-1.5.4/b-colvis-1.5.4/b-html5-1.5.4/b-print-1.5.4/datatables.min.js"></script>
    
		<script type="text/javascript" language="javascript" class="init">
   		$(document).ready(function() {
			$('#ensembleStatistics').DataTable({
        		ordering:		false,
				scrollY:        true,
				scrollX:        true,
				scrollCollapse: true,
				paging:         false
			});
		});
		</script>
		</html>
		"""
		file.write(next_str)

		file.close()

	def html_dual_123(self):
		file = open(os.path.join(self.scratch, 'forHTML', 'dual_123.html'), u"w")

		html_string = u"""
		<!DOCTYPE html>
		<html>
		<head>
		<style>
		body {font-family: "Lato", sans-serif;}
		/* Style the tab */
		div.tab {
			overflow: hidden;
			border: 1px solid #ccc;
			background-color: #f1f1f1;
		}
		/* Style the buttons inside the tab */
		div.tab button {
			background-color: inherit;
			float: left;
			border: none;
			outline: none;
			cursor: pointer;
			padding: 14px 16px;
			transition: 0.3s;
			font-size: 17px;
		}
		/* Change background color of buttons on hover */
		div.tab button:hover {
			background-color: #ddd;
		}
		/* Create an active/current tablink class */
		div.tab button.active {
			background-color: #ccc;
		}
		/* Style the tab content */
		.tabcontent {
			display: none;
			padding: 6px 12px;
			border: 1px solid #ccc;
			-webkit-animation: fadeEffect 1s;
			animation: fadeEffect 1s;
			border-top: none;
		}
		/* Fade in tabs */
		@-webkit-keyframes fadeEffect {
			from {opacity: 0;}
			to {opacity: 1;}
		}
		@keyframes fadeEffect {
			from {opacity: 0;}
			to {opacity: 1;}
		}
		table {
			font-family: arial, sans-serif;
			border-collapse: collapse;
			width: 100%;
		}
		td, th {
			border: 1px solid #dddddd;
			text-align: left;
			padding: 8px;
		}
		tr:nth-child(odd) {
			background-color: #dddddd;
		}
		div.gallery {
			margin: 5px;
			border: 1px solid #ccc;
			float: left;
			width: 180px;
		}
		div.gallery:hover {
			border: 1px solid #777;
		}
		div.gallery img {
			width: 100%;
			height: auto;
		}
		div.desc {
			padding: 15px;
			text-align: center;
		}
		</style>
		</head>
		<body>

		<p></p>

		<div class="tab">
		  <button class="tablinks" onclick="openTab(event, 'Overview')" id="defaultOpen">Main Analysis</button>
		  <button class="tablinks" onclick="openTab(event, 'Visualization')">Decision Tree Analysis</button>
		  <button class="tablinks" onclick="openTab(event, 'ThirdPage')">Ensemble Model</button>
		</div>

		<div id="Overview" class="tabcontent">
		  <iframe src="html1folder/html1.html" style="height:100vh; width:100%; border: hidden;" ></iframe>
		</div>

		<div id="Visualization" class="tabcontent">
		  <iframe src="html2folder/html2.html" style="height:100vh; width:100%; border: hidden;" ></iframe>
		</div>

		<div id="ThirdPage" class="tabcontent">
		  <iframe src="html4folder/html4.html" style="height:100vh; width:100%; border: hidden;" ></iframe>
		</div>

		<script>
		function openTab(evt, tabName) {
			var i, tabcontent, tablinks;
			tabcontent = document.getElementsByClassName("tabcontent");
			for (i = 0; i < tabcontent.length; i++) {
				tabcontent[i].style.display = "none";
			}
			tablinks = document.getElementsByClassName("tablinks");
			for (i = 0; i < tablinks.length; i++) {
				tablinks[i].className = tablinks[i].className.replace(" active", "");
			}
			document.getElementById(tabName).style.display = "block";
			evt.currentTarget.className += " active";
		}
		// Get the element with id="defaultOpen" and click on it
		document.getElementById("defaultOpen").click();
		</script>

		</body>
		</html>
		"""

		file.write(html_string)
		file.close()

		return "dual_123.html"

	def html_dual_12(self):
		file = open(os.path.join(self.scratch, 'forHTML', 'dual_12.html'), u"w")

		html_string = u"""
		<!DOCTYPE html>
		<html>
		<head>
		<style>
		body {font-family: "Lato", sans-serif;}
		/* Style the tab */
		div.tab {
			overflow: hidden;
			border: 1px solid #ccc;
			background-color: #f1f1f1;
		}
		/* Style the buttons inside the tab */
		div.tab button {
			background-color: inherit;
			float: left;
			border: none;
			outline: none;
			cursor: pointer;
			padding: 14px 16px;
			transition: 0.3s;
			font-size: 17px;
		}
		/* Change background color of buttons on hover */
		div.tab button:hover {
			background-color: #ddd;
		}
		/* Create an active/current tablink class */
		div.tab button.active {
			background-color: #ccc;
		}
		/* Style the tab content */
		.tabcontent {
			display: none;
			padding: 6px 12px;
			border: 1px solid #ccc;
			-webkit-animation: fadeEffect 1s;
			animation: fadeEffect 1s;
			border-top: none;
		}
		/* Fade in tabs */
		@-webkit-keyframes fadeEffect {
			from {opacity: 0;}
			to {opacity: 1;}
		}
		@keyframes fadeEffect {
			from {opacity: 0;}
			to {opacity: 1;}
		}
		table {
			font-family: arial, sans-serif;
			border-collapse: collapse;
			width: 100%;
		}
		td, th {
			border: 1px solid #dddddd;
			text-align: left;
			padding: 8px;
		}
		tr:nth-child(odd) {
			background-color: #dddddd;
		}
		div.gallery {
			margin: 5px;
			border: 1px solid #ccc;
			float: left;
			width: 180px;
		}
		div.gallery:hover {
			border: 1px solid #777;
		}
		div.gallery img {
			width: 100%;
			height: auto;
		}
		div.desc {
			padding: 15px;
			text-align: center;
		}
		</style>
		</head>
		<body>

		<p></p>

		<div class="tab">
		  <button class="tablinks" onclick="openTab(event, 'Overview')" id="defaultOpen">Main Analysis</button>
		  <button class="tablinks" onclick="openTab(event, 'Visualization')">Decision Tree Analysis</button>
		</div>

		<div id="Overview" class="tabcontent">
		  <iframe src="html1folder/html1.html" style="height:100vh; width:100%; border: hidden;" ></iframe>
		</div>

		<div id="Visualization" class="tabcontent">
		  <iframe src="html2folder/html2.html" style="height:100vh; width:100%; border: hidden;" ></iframe>
		</div>

		<script>
		function openTab(evt, tabName) {
			var i, tabcontent, tablinks;
			tabcontent = document.getElementsByClassName("tabcontent");
			for (i = 0; i < tabcontent.length; i++) {
				tabcontent[i].style.display = "none";
			}
			tablinks = document.getElementsByClassName("tablinks");
			for (i = 0; i < tablinks.length; i++) {
				tablinks[i].className = tablinks[i].className.replace(" active", "");
			}
			document.getElementById(tabName).style.display = "block";
			evt.currentTarget.className += " active";
		}
		// Get the element with id="defaultOpen" and click on it
		document.getElementById("defaultOpen").click();
		</script>

		</body>
		</html>
		"""

		file.write(html_string)
		file.close()

		return "dual_12.html"

	### For Predict_Phenotype App	
	def html_report_3(self, missingGenomes, phenotype):
		"""
		does: creates an .html file that makes the first report (second app).
		"""
		file = open(os.path.join(self.scratch, 'forSecHTML', 'html3.html'), u"w")

		html_string = u"""
		<!DOCTYPE html>
		<html>

		<head>
		
		<link href="https://cdnjs.cloudflare.com/ajax/libs/twitter-bootstrap/4.1.3/css/bootstrap.css" rel="stylesheet">

		<style>
		table, th, td , tr{
			text-align: center;
		}
		</style>

		</head>
		<body>
		<div class="container">

		<h1 style="text-align:center;"> Predicting """ + phenotype + """ of Genomes</h1>

		<p> Missing genomes are listed below (ie. these were included in your excel / pasted file but are not present in the workspace).
		In the event that there were missing genomes they were excluded  </p>
		<p> The missing genomes are: """ + str(missingGenomes) + """ </p>
		
		<br>

		"""
		file.write(html_string)


		sec_string = u"""

		<h1 style="text-align:left;">Prediction Results</h1>

		<!-- <h2>Maybe we can add some more text here later?</h2> -->
		<!--<p>How to create side-by-side images with the CSS float property:</p> -->

		<p>  Here is a simple table that shows the prediction for each sample and the probability of that prediction being correct </p>

		<p> Ways of improving predictions </p>
		<ul>
			<li>Gather more data</li>
			<li>Tune Hyperparameters (additional options in Build Classifier App)</li>
		</ul> 

		"""
		file.write(sec_string)

		
		another_file = open(os.path.join(self.scratch, 'forSecHTML', 'html3folder', 'results.html'), u"r")
		all_str = another_file.read()
		another_file.close()

		file.write(all_str)
		

		next_str= u"""
		</div>
		</body>

		<script type="text/javascript" src="https://code.jquery.com/jquery-3.3.1.js"></script>
		<script type="text/javascript" src="https://cdn.datatables.net/v/bs4-4.1.1/jq-3.3.1/dt-1.10.18/b-1.5.4/b-colvis-1.5.4/b-html5-1.5.4/b-print-1.5.4/datatables.min.js"></script>
    
		<script type="text/javascript" language="javascript" class="init">
   		$(document).ready(function() {
			$('#results').DataTable();
		});
		</script>
		</html>
		"""

		file.write(next_str)

		file.close()

		return "html3.html"

	def html_nodual(self, location):

		if location == "forHTML":
			file = open(os.path.join(self.scratch, 'forHTML', 'nodual.html'), u"w")
		elif location == "forSecHTML":
			file = open(os.path.join(self.scratch, 'forSecHTML', 'nodual.html'), u"w")
		else:
			file = open(os.path.join(self.scratch, 'forZeroHTML', 'nodual.html'), u"w")

		html_string = u"""
		<!DOCTYPE html>
		<html>
		<head>
			<style>
			body {font-family: "Lato", sans-serif;}
			/* Style the tab */
			div.tab {
				overflow: hidden;
				border: 1px solid #ccc;
				background-color: #f1f1f1;
			}
			/* Style the buttons inside the tab */
			div.tab button {
				background-color: inherit;
				float: left;
				border: none;
				outline: none;
				cursor: pointer;
				padding: 14px 16px;
				transition: 0.3s;
				font-size: 17px;
			}
			/* Change background color of buttons on hover */
			div.tab button:hover {
				background-color: #ddd;
			}
			/* Create an active/current tablink class */
			div.tab button.active {
				background-color: #ccc;
			}
			/* Style the tab content */
			.tabcontent {
				display: none;
				padding: 6px 12px;
				border: 1px solid #ccc;
				-webkit-animation: fadeEffect 1s;
				animation: fadeEffect 1s;
				border-top: none;
			}
			/* Fade in tabs */
			@-webkit-keyframes fadeEffect {
				from {opacity: 0;}
				to {opacity: 1;}
			}
			@keyframes fadeEffect {
				from {opacity: 0;}
				to {opacity: 1;}
			}
			table {
				font-family: arial, sans-serif;
				border-collapse: collapse;
				width: 100%;
			}
			td, th {
				border: 1px solid #dddddd;
				text-align: left;
				padding: 8px;
			}
			tr:nth-child(odd) {
				background-color: #dddddd;
			}
			div.gallery {
				margin: 5px;
				border: 1px solid #ccc;
				float: left;
				width: 180px;
			}
			div.gallery:hover {
				border: 1px solid #777;
			}
			div.gallery img {
				width: 100%;
				height: auto;
			}
			div.desc {
				padding: 15px;
				text-align: center;
			}
		</style>
		</head>
		<body>

			<p></p>

			<div class="tab">
			  <button class="tablinks" onclick="openTab(event, 'Overview')" id="defaultOpen">Main Analysis</button>
		  </div>
		"""
		file.write(html_string)

		if location == "forHTML":
			next_str = u"""
			  <div id="Overview" class="tabcontent">
				  <iframe src="html1folder/html1.html" style="height:100vh; width:100%; border: hidden;" ></iframe>
			  </div>
			  """
			file.write(next_str)
		elif location == "forSecHTML" :
			next_str = u"""
			  <div id="Overview" class="tabcontent">
				  <iframe src="html3.html" style="height:100vh; width:100%; border: hidden;" ></iframe>
			  </div>
			  """         
			file.write(next_str)  
		else:
			next_str = u"""
			  <div id="Overview" class="tabcontent">
				  <iframe src="html0.html" style="height:100vh; width:100%; border: hidden;" ></iframe>
			  </div>
			  """         
			file.write(next_str)  


		next_str = u"""
		  <script>
			function openTab(evt, tabName) {
				var i, tabcontent, tablinks;
				tabcontent = document.getElementsByClassName("tabcontent");
				for (i = 0; i < tabcontent.length; i++) {
					tabcontent[i].style.display = "none";
				}
				tablinks = document.getElementsByClassName("tablinks");
				for (i = 0; i < tablinks.length; i++) {
					tablinks[i].className = tablinks[i].className.replace(" active", "");
				}
				document.getElementById(tabName).style.display = "block";
				evt.currentTarget.className += " active";
			}
				// Get the element with id="defaultOpen" and click on it
				document.getElementById("defaultOpen").click();
			</script>

		</body>
		</html>
		"""
		file.write(next_str)
		file.close()

		return "nodual.html"