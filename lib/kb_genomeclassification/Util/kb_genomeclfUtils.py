# -*- coding: utf-8 -*-
#BEGIN_HEADER
# The header block is where all import statments should live

from __future__ import division

import os
import re
import uuid
import sys
import codecs
import graphviz
import StringIO
import xlrd

import pickle
import numpy as np
import pandas as pd
import seaborn as sns

from io import open

import itertools
from itertools import izip

#classifier models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier

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
from biokbase.workspace.client import Workspace as workspaceService



class kb_genomeclfUtils(object):
	"""docstring for ClassName"""
	def __init__(self, config):

		self.workspaceURL = config['workspaceURL']
		self.scratch = config['scratch']
		self.callback_url = config['callback_url']

		self.ctx = config['ctx']

		self.dfu = DataFileUtil(self.callback_url)
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
		if params.get('list_name'):
			#checks if empty string bool("") --> False
			print ("taking this path rn")
			toEdit_all_classifications = self.incaseList_Names(params.get('list_name'))
			listOfNames, all_classifications = self.intake_method(toEdit_all_classifications)
			all_attributes, master_Role = self.get_wholeClassification(listOfNames, current_ws)
		else:
			file_path = self._download_shock(params.get('shock_id'))
			listOfNames, all_classifications = self.intake_method(just_DF = pd.read_excel(file_path))
			all_attributes, master_Role = self.get_wholeClassification(listOfNames, current_ws)

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

		self.createHoldingDirs()

		ctx = self.ctx
		token = ctx['token']

		classifier_type = params.get('classifier')
		global_target = params.get('phenotypeclass')
		classifier_name = params.get('classifier_out')

		folderhtml1 = "html1folder/"
		folderhtml2 = "html2folder/"

		train_index = []
		test_index = []

		splits = 2 #10


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
				'print_cfm' : True
				}

		#RUNNING the classifiers depending on classifier_type

		if classifier_type == u"run_all":

			listRunAll = ['KNeighborsClassifier', 'GaussianNB', 'LogisticRegression', 'DecisionTreeClassifier', 'SVM', 'NeuralNetwork']

			for run in listRunAll:
				classifierTest_params['classifier'] = self.whichClassifier(run)
				classifierTest_params['classifier_type'] = run
				classifierTest_params['classifier_name'] = classifier_name + u'_' + run
				classifierTest_params['htmlfolder'] = folderhtml1

				self.classifierTest(classifierTest_params)

			best_classifier_str = self.to_HTML_Statistics(class_list, classifier_name)
			#best_classifier_str = classifier_name+u"_LogisticRegression"
			best_classifier_type = best_classifier_str[classifier_name.__len__() + 1:] #extract just the classifier_type aka. "LogisticRegression" from "myName_LogisticRegression"
			
			#to create another "best in html2"
			
			classifierTest_params['classifier'] = self.whichClassifier(best_classifier_type)
			classifierTest_params['classifier_type'] = best_classifier_type
			classifierTest_params['classifier_name'] = classifier_name+u"_" + best_classifier_type
			classifierTest_params['htmlfolder'] = folderhtml2
			self.classifierTest(classifierTest_params)

			#self.classifierTest(ctx, current_ws, self.whichClassifier(best_classifier_type),best_classifier_type, classifier_name+u"_" + best_classifier_type, my_mapping, master_Role, splits, train_index, test_index, all_attributes, all_classifications, class_list, folderhtml2, True)


			self.html_report_1(global_target, classifier_type, classifier_name, best_classifier_str= best_classifier_str)

			#classifierTest_params['classifier_type'] = "DecisionTreeClassifier"
			classifierTest_params['classifier_name'] = classifier_name
			classifierTest_params['htmlfolder'] = folderhtml2
			self.tune_Decision_Tree(classifierTest_params, best_classifier_str)
			
			self.html_report_2(global_target, classifier_name, best_classifier_str)
			htmloutput_name = self.html_dual_12()

		elif classifier_type == u"DecisionTreeClassifier":

			classifierTest_params['classifier'] = self.whichClassifier(classifier_type)
			self.classifierTest(classifierTest_params)

			self.to_HTML_Statistics(class_list, classifier_name)
			
			self.html_report_1(global_target, classifier_type, classifier_name)


			classifierTest_params['htmlfolder'] = folderhtml2
			self.tune_Decision_Tree(classifierTest_params)
			
			self.html_report_2(global_target, classifier_name)
			htmloutput_name = self.html_dual_12()

		else:

			classifierTest_params['classifier'] = self.whichClassifier(classifier_type)
			self.classifierTest(classifierTest_params)

			self.to_HTML_Statistics(class_list, classifier_name)
			self.html_report_1(global_target, classifier_type, classifier_name)
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

		base64str = str(classifier_object[0]['data']['classifier_data'])
		master_Role = classifier_object[0]['data']['attribute_data']
		my_mapping = classifier_object[0]['data']['class_list_mapping']

		after_classifier = pickle.loads(codecs.decode(base64str.encode(), "base64"))

		if params.get('list_name'):
			#checks if empty string bool("") --> False
			toEdit_all_classifications = self.incaseList_Names(params.get('list_name'), for_predict = True)
			listOfNames = self.intake_method(toEdit_all_classifications, for_predict = True)
			all_attributes = self.get_wholeClassification(listOfNames, current_ws, master_Role = master_Role ,for_predict = True)
		else:
			file_path = self._download_shock(params.get('shock_id'))
			listOfNames, all_classifications = self.intake_method(just_DF = pd.read_excel(file_path))
			all_attributes = self.get_wholeClassification(listOfNames, current_ws, master_Role = master_Role ,for_predict = True)

		#PREDICTIONS on new data that needs to be classified
		after_classifier_result = after_classifier.predict(all_attributes)

		after_classifier_result_forDF = []

		for current_result in after_classifier_result:
			after_classifier_result_forDF.append(my_mapping.keys()[my_mapping.values().index(current_result)])

		"""
		print "I'm printing after_classifier_result_forDF"
		print (after_classifier_result_forDF)

		print "I'm printing all_attributes.index"
		print (all_attributes.index)
		"""
		
		after_classifier_df = pd.DataFrame(after_classifier_result_forDF, index=all_attributes.index, columns=[target])

		#create a column for the probability of a prediction being accurate
		allProbs = after_classifier.predict_proba(all_attributes)
		maxEZ = np.amax(allProbs, axis=1)
		maxEZ_df = pd.DataFrame(maxEZ, index=all_attributes.index, columns=["probability"])

		predict_table_pd = pd.concat([after_classifier_df, maxEZ_df], axis=1)
		predict_table_pd.to_html(os.path.join(self.scratch, 'forSecHTML', 'html3folder', 'results.html'))

		#you can also save down table as text file or csv
		"""
		#txt
		np.savetxt(r'/kb/module/work/tmp/np.txt', predict_table_pd.values, fmt='%d')

		#csv
		predict_table_pd.to_csv(r'/kb/module/work/tmp/pandas.txt', header=None, index=None, sep=' ', mode='a')
		"""
		self.html_report_3()
		htmloutput_name = self.html_nodual("forSecHTML")

		return htmloutput_name

	def makeHtmlReport(self, htmloutput_name, current_ws, which_report):
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

		output_directory = os.path.join(self.scratch, saved_html)
		report_shock_id = self.dfu.file_to_shock({'file_path': output_directory,'pack': 'zip'})['shock_id']

		htmloutput = {
		'description' : 'htmloutuputdescription',
		'name' : htmloutput_name,
		'label' : 'htmloutputlabel',
		'shock_id': report_shock_id
		}

		"""output_zip_files.append({'path': os.path.join(read_file_path, file),
																				 'name': file,
																				 'label': label,
																				 'description': desc})"""

		report_params = {'message': '',
			 'workspace_name': current_ws,#params.get('input_ws'),
			 #'objects_created': objects_created,
			 #'file_links': output_zip_files,
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

		print "before closing"

		tem_file.close()

		if not for_predict:
			print "I'm inside the if not for_predict"
			my_workPD = pd.read_csv(os.path.join(self.scratch, 'trialoutput.txt'), delimiter="\s+")
			#print my_workPD
		else: 
			my_workPD = pd.read_csv(os.path.join(self.scratch, 'trialoutput.txt'))

		os.remove(os.path.join(self.scratch, 'trialoutput.txt'))

		return my_workPD

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

		print "Below is my_all_classifications"

		#print my_all_classifications

		if not for_predict:
			return my_all_classifications.index, my_all_classifications
		else:
			return my_all_classifications.index


	def get_wholeClassification(self, listOfNames, current_ws, master_Role = None, for_predict = False):
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


		name_and_roles = {}

		for current_gName in listOfNames:
			listOfFunctionalRoles = []
			try:
				functionList = self.ws_client.get_objects([{'workspace':current_ws, 'name':current_gName}])[0]['data']['cdss']
				for function in range(len (functionList)):
					if str(functionList[function]['functions'][0]).lower() != 'hypothetical protein':
						listOfFunctionalRoles.append(str(functionList[function]['functions'][0]))

			except:
				functionList = self.ws_client.get_objects([{'workspace':current_ws, 'name':current_gName}])[0]['data']['features']
				for function in range(len (functionList)):
					if str(functionList[function]['function']).lower() != 'hypothetical protein':
						listOfFunctionalRoles.append(str(functionList[function]['function']))

			name_and_roles[current_gName] = listOfFunctionalRoles

			print "I have arrived inside the desired for loop!!"

		if not for_predict:
			master_pre_Role = list(itertools.chain(*name_and_roles.values()))
			master_Role = list(set(master_pre_Role))


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
			dirs_names = ['pics', 'dotFolder', 'forHTML', 'forHTML/html1folder', 'forHTML/html2folder', 'forHTML/forDATA']
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
			return svm.LinearSVC(random_state=0)
		elif name == u"NeuralNetwork":
			return MLPClassifier(random_state=0)
		else:
			return u"ERROR THIS SHOULD NOT HAVE REACHED HERE"

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

		if print_cfm:
			print classifier_name
			self.list_name.extend([classifier_name])
		train_score = np.zeros(splits)
		validate_score = np.zeros(splits)
		matrix_size = class_list.__len__()

		cnf_matrix = np.zeros(shape=(matrix_size, matrix_size))
		cnf_matrix_f = np.zeros(shape=(matrix_size, matrix_size))
		for c in xrange(splits):
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
			for i in xrange(len(cnf)):
				for j in xrange(len(cnf)):
					cnf_matrix[i][j] += cnf[i][j]
					cnf_matrix_f[i][j] += cnf_f[i][j]

		if print_cfm:
			pickle_out = open(os.path.join(self.scratch, 'forHTML', 'forDATA', unicode(classifier_name) + u".pickle"), u"wb")

			#pickle_out = open("/kb/module/work/tmp/" + str(self.classifier_name) + ".pickle", "wb")


			pickle.dump(classifier.fit(all_attributes, all_classifications), pickle_out, protocol = 2)
			pickle_out.close()


			current_pickle = pickle.dumps(classifier.fit(all_attributes, all_classifications), protocol=0)
			pickled = codecs.encode(current_pickle, "base64").decode()


			"""

			with open(u"/kb/module/work/tmp/" + unicode(classifier_name) + u".txt", u"w") as f:
				for line in pickled:
					f.write(line)
			"""

			#pickled = "this is what the pickled string would be"

			print ""
			print "This is printing out the classifier_object that needs to be saved down dump"
			print ""

			
			classifier_object = {
			'classifier_id' : '',
			'classifier_type' : classifier_type, # Neural network
			'classifier_name' : classifier_name,
			'classifier_data' : pickled,
			'classifier_description' : 'this is my description',
			'lib_name' : 'sklearn',
			'attribute_type' : 'functional_roles',
			'number_of_attributes' : class_list.__len__(),
			'attribute_data' : master_Role,#["this is where master_role would go", "just a list"],#master_Role, #master_Role,
			'class_list_mapping' : my_mapping, #{} my_mapping, #my_mapping,
			'number_of_genomes' : all_attributes.shape[0],
			'training_set_ref' : ''
			}

			#print classifier_object

			#Saving the Classifier object
	
			obj_save_ref = self.ws_client.save_objects({'workspace': current_ws,
														  'objects':[{
														  'type': 'KBaseClassifier.GenomeClassifier',
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

		AN = TN + FP
		AP = TN + FN
		PN = TN + FN
		PP = TP + FP
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
		sns_plot = sns.heatmap(cm, annot=True, ax = ax, cmap=u"Blues"); #annot=True to annotate cells
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

	def to_HTML_Statistics(self, class_list, classifier_name, known = "", additional = False):
		"""
		args:
		---additional is a boolean and is used to indicate if this method is being called to make html2
		does:
		---the statistics that were calculated and stored into lists are converted into a dataframe table --> html page
		return:
		---N/A but instead creates an html file in tmp
		"""


		#self.counter = self.counter + 1

		if not additional:

			print u"I am inside not additional"

			statistics_dict = {}

			#print(self.list_name)
			#print(self.list_statistics)

			for i, j in izip(self.list_name, self.list_statistics):
				statistics_dict[i] = j

			data = statistics_dict

			if class_list.__len__() == 3:
				my_index = [class_list[0], u'Accuracy:', u'Precision:', u'Recall:', u'F1 score::', None, class_list[1], u'Accuracy:',
						u'Precision:', u'Recall:', u'F1 score::', None, class_list[2], u'Accuracy:', u'Precision:', u'Recall:',
						u'F1 score::', u'Average F1']

			if class_list.__len__() == 2:
				my_index = [u'Accuracy:', u'Precision:', u'Recall:', u'F1 score::']

			df = pd.DataFrame(data, index=my_index)

			df.to_html(os.path.join(self.scratch, 'forHTML', 'html1folder','newStatistics.html'))

			df['Max'] = df.idxmax(1)
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

			if class_list.__len__() == 3:
				my_index = [class_list[0], u'Accuracy:', u'Precision:', u'Recall:', u'F1 score::', None, class_list[1], u'Accuracy:',
						u'Precision:', u'Recall:', u'F1 score::', None, class_list[2], u'Accuracy:', u'Precision:', u'Recall:',
						u'F1 score::', u'Average F1']

			if class_list.__len__() == 2:
				my_index = [u'Accuracy:', u'Precision:', u'Recall:', u'F1 score::']

			df = pd.DataFrame(data, index=my_index)
			df.to_html(os.path.join(self.scratch, 'forHTML', 'html2folder', 'postStatistics.html'))

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
				'print_cfm' : True
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

		self.parse_lookNice(pass_name, class_list)

	def parse_lookNice(self, name, class_list):
		"""
		args:
		---name is a string that is what you want the DecisionTree image saved as
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

		first_fix = re.sub(ur'(\w\s\[label="[\w\s.,:\/()-]+)<=([\w\s.\[\]=,]+)("] ;)',
						   ur'\1 (Absent)" , color="0.650 0.200 1.000"] ;', new_allStr)
		second_fix = re.sub(ur'(\w\s\[label=")(.+?class\s=\s)', ur'\1', first_fix)

		# nominal fixes like color and shape
		third_fix = re.sub(ur'shape=box] ;', ur'shape=Mrecord] ; node [style=filled];', second_fix)

		if class_list.__len__() == 3:
			fourth_fix = re.sub(ur'(\w\s\[label="%s")' % class_list[0], ur'\1, color = "0.5176 0.2314 0.9020"', third_fix)
			fifth_fix = re.sub(ur'(\w\s\[label="%s")' % class_list[1], ur'\1, color = "0.5725 0.6118 1.0000"', fourth_fix)
			sixth_fix = re.sub(ur'(\w\s\[label="%s")' % class_list[2], ur'\1, color = "0.5804 0.8824 0.8039"', fifth_fix)
			f = open(os.path.join(self.scratch, 'dotFolder', 'niceTree.dot'), u"w")
			f.write(sixth_fix)
			f.close()

			os.system(u'dot -Tpng ' + os.path.join(self.scratch, 'dotFolder', 'niceTree.dot') + ' >  '+ os.path.join(self.scratch, 'forHTML', 'html2folder', name + u'.png '))

		if class_list.__len__() == 2:
			fourth_fix = re.sub(ur'(\w\s\[label="%s")' % class_list[0], ur'\1, color = "0.5176 0.2314 0.9020"', third_fix)
			fifth_fix = re.sub(ur'(\w\s\[label="%s")' % class_list[1], ur'\1, color = "0.5725 0.6118 1.0000"', fourth_fix)
			f = open(os.path.join(self.scratch, 'dotFolder', 'niceTree.dot'), u"w")
			f.write(fifth_fix)
			f.close()

			os.system(u'dot -Tpng ' + os.path.join(self.scratch, 'dotFolder', 'niceTree.dot') + ' >  '+ os.path.join(self.scratch, 'forHTML', 'html2folder', name + u'.png '))


	### Extra methods being used 
	def _make_dir(self):
		dir_path = os.path.join(self.scratch, str(uuid.uuid4()))
		os.mkdir(dir_path)

		return dir_path

	def _download_shock(self, shock_id):
		"""
		does:
		---using kbase dfu tool to allow users to insert excel files 
		"""
		dir_path = self._make_dir()

		file_path = self.dfu.shock_to_file({'shock_id': shock_id,
											'file_path': dir_path})['file_path']

		return file_path

	#### HTML templates below ####

	### For Build_Classifier App
	def html_report_1(self, global_target, classifier_type, classifier_name, best_classifier_str = None):
		"""
		does: creates an .html file that makes the frist report (first app).
		"""
		file = open(os.path.join(self.scratch, 'forHTML', 'html1folder', 'html1.html'), u"w")

		html_string = u"""
		<!DOCTYPE html>
		<html>
		<head>
		<style>
		table, th, td {
			border: 1px solid black;
			border-collapse: collapse;
		}

		* {
			box-sizing: border-box;
		}

		.column {
			float: left;
			width: 50%;
			padding: 10px;
		}

		/* Clearfix (clear floats) */
		.row::after {
			content: "";
			clear: both;
			display: table;
		}
		</style>
		</head>
		<body>

		<h1 style="text-align:center;">""" + global_target + """ Classifier</h1> """

		file.write(html_string)

		if classifier_type == u"run_all":
			
			next_str = u"""
			<p style="text-align:center; font-size:160%;">  Prediction of respiration type based on classifiers depicted in the form of confusion matrices*. 
			A.) K-Nearest-Neighbors Classifier,  
			B.) Logistic Regression Classifier,
			C.) Naive Gaussian Bayes Classifier,
			D.) Linear Support Vector Machine (SVM) Classifier,
			E.) Decision Tree Classifier,  
			F.) Neural Network Classifier</p>
			<h2> Disclaimer:No feature selection and parameter optimization was not done</h2>
			"""

			file.write(next_str)

		else :

			next_str = u"""
			<p style="text-align:center; font-size:160%;">  Prediction of respiration type based on classifiers depicted in the form of confusion matrices*. <br/>
			 A.) """ + classifier_type + """</p>  
			<h2> Disclaimer:No feature selection and parameter optimization was not done</h2>
			"""

			file.write(next_str)

		next_str = u"""
		<p style="font-size:110%;"> 
		* A confusion matrix is a table that is used to describe the performance of a classifier
		on a set of test data for which the true values are known - showing the comparision between the predicted labels and true labels. (In our case we used 
		K-fold Cross Validation and the below confusion matrices represent the "average" of k-folds.) 
		The number in each cell of the confusion matrix is the percentage of samples with a certain label. Confusion matrices are read by row. 
		For example: ___number %___ of the ___row[0] lable___ were predicted as being ___column[0] lable___.
		A strong classifier is one that has a central diagonal with the highest percentages, meaning that the majority of the predicted labels match the true label.
		</p>,

		<br/>

		<p style="font-size:110%;">**Furthermore each classifer is available to download via the Download link in a .pickle format (sklearn classifiers)</p>
		"""

		file.write(next_str)

		if classifier_type == u"run_all":
			next_str = u"""
		<div class="row">
		  <div class="column">
			  <p style="text-align:left; font-size:160%;">A.) K-Nearest-Neighbors Classifier <a href="../forDATA/""" + classifier_name + """_KNeighborsClassifier.pickle" download> (Download) </a> </p>
			<img src=" """+ classifier_name +"""_KNeighborsClassifier.png" alt="Snow" style="width:100%">
			  <!-- <figcaption>Fig.1 - Trulli, Puglia, Italy.</figcaption> -->
		  </div>
		  <div class="column">
			  <p style="text-align:left; font-size:160%;">B.) Logistic Regression Classifier <a href="../forDATA/""" + classifier_name + """_LogisticRegression.pickle" download> (Download) </a> </p>
			<img src=" """+ classifier_name +"""_LogisticRegression.png" alt="Snow" style="width:100%">
		  </div>
		</div>

		<div class="row">
		  <div class="column">
			  <p style="text-align:left; font-size:160%;">C.) Naive Gaussian Bayes Classifier <a href="../forDATA/""" + classifier_name + """_GaussianNB.pickle" download> (Download) </a> </p>
			<img src=" """+ classifier_name +"""_GaussianNB.png" alt="Snow" style="width:100%">
		  </div>
		  <div class="column">
			  <p style="text-align:left; font-size:160%;">D.) Linear Support Vector Machine (SVM) Classifier <a href="../forDATA/""" + classifier_name + """_SVM.pickle" download> (Download) </a> </p>
			<img src=" """+ classifier_name +"""_SVM.png" alt="Snow" style="width:100%">
		  </div>
		</div>

		<div class="row">
		  <div class="column">
			  <p style="text-align:left; font-size:160%;">E.) Decision Tree Classifier <a href="../forDATA/""" + classifier_name + """_DecisionTreeClassifier.pickle" download> (Download) </a> </p>
			<img src=" """+ classifier_name +"""_DecisionTreeClassifier.png" alt="Snow" style="width:100%">
		  </div>
		  <div class="column">
			  <p style="text-align:left; font-size:160%;">F.) Neural Network Classifier <a href="../forDATA/""" + classifier_name + """_NeuralNetwork.pickle" download> (Download) </a> </p>
			<img src=" """+ classifier_name +"""_NeuralNetwork.png" alt="Snow" style="width:100%">
		  </div>
		</div>
			"""
			file.write(next_str)

			next_str = u"""
			<p style="font-size:160%;">Comparison of statistics in the form of Accuracy, Precision, Recall and F1 Score calculated against the confusion matrices of respiration type for the classifiers</p>
			<p style="font-size:100%;">Defintion of key statistics: Accuracy - how often is the classifier correct, Precision - when predition is positive how often is it correct, 
			Recall - when the condition is correct how often is it correct, F1 Score - This is a weighted average of recall and precision </p>               
			"""
			file.write(next_str)

			another_file = open(os.path.join(self.scratch, 'forHTML', 'html1folder', 'newStatistics.html'), u"r")
			all_str = another_file.read()
			another_file.close()

			file.write(all_str)

			next_str = u"""
			<p style="text-align:center; font-size:100%;">  Based on these results it would be in your best interest to use the """ + unicode(best_classifier_str) + """ as your model as
			it produced the strongest F1 score </p>
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
			<p style="font-size:160%;">Comparison of statistics in the form of Accuracy, Precision, Recall and F1 Score calculated against the confusion matrices of respiration type for the classifiers</p>
			<p style="font-size:100%;">Defintion of key statistics: Accuracy - how often is the classifier correct, Precision - when predition is positive how often is it correct, 
			Recall - when the condition is correct how often is it correct, F1 Score - This is a weighted average of recall and precision </p>            
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
		<a href="../html2folder/html2.html">Second html page</a>
		"""

		#file.write(next_str)

		file.close()

	def html_report_2(self, global_target, classifier_name, best_classifier_str = None):
		"""
		does: creates an .html file that makes the second report (first app).
		"""
		file = open(os.path.join(self.scratch, 'forHTML', 'html2folder', 'html2.html'), u"w")

		html_string = u"""
		<!DOCTYPE html>
		<html>
		<head>
		<style>
		table, th, td {
			border: 1px solid black;
			border-collapse: collapse;
		}

		* {
			box-sizing: border-box;
		}

		.column {
			float: left;
			width: 50%;
			padding: 10px;
		}

		/* Clearfix (clear floats) */
		.row::after {
			content: "";
			clear: both;
			display: table;
		}
		</style>
		</head>
		<body>

		<h1 style="text-align:center;">""" + global_target + """ - Decision Tree Tuning</h1>

		<!-- <h2>Maybe we can add some more text here later?</h2> -->
		<!--<p>How to create side-by-side images with the CSS float property:</p> -->

		<p style="text-align:center; font-size:160%;">  Comparison of Accuracy between Training versus Testing data sets based on the Gini Criterion and the Entropy Criterion for 11 levels of Tree Depth </p>
		<p style="text-align:center; font-size:100%;">  (Below is the training and testing accuracy at each tree depth. The tuning parameter choosen was criterion which measures the quality of a split. The criteria were "gini" for the Gini impurity and "entropy" for the information gain.) </p>
		"""

		file.write(html_string)

		next_str = u"""

		<div class="row">
		  <div class="column">
			  <p style="text-align:left; font-size:160%;">Training vs Testing Score on Gini Criterion </p>
			<img src=" """+ classifier_name +"""_gini_depth-met.png" alt="Snow" style="width:100%">
			  <!-- <figcaption>Fig.1 - Trulli, Puglia, Italy.</figcaption> -->
		  </div>
		  <div class="column">
			  <p style="text-align:left; font-size:160%;">Training vs Testing Score on Entropy Criterion</p>
			<img src=" """+ classifier_name +"""_entropy_depth-met.png" alt="Snow" style="width:100%">
		  </div>
		</div>
		"""

		file.write(next_str)

		if best_classifier_str == None :
			next_str = u"""<p style="text-align:center; font-size:160%;">  Comparison of tuned Gini and Entropy based Decision Tree Classifiers depicted in the form of confusion matrices. 
			A.) Decision Tree Classifier 
			B.) Decision Tree Classifier-Gini 
			C.) Decision Tree Classifier-Entropy 
			<p style="text-align:center; font-size:100%;">  The original Decision Tree Classifier model was chosen as a base comparision </p>

			<div class="row">
			  <div class="column">
				  <p style="text-align:left; font-size:160%;">A.) Decision Tree Classifier <a href="../forDATA/""" + classifier_name + """.pickle" download> (Download) </a> </p>
				<img src=" """+ classifier_name +""".png" alt="Snow" style="width:100%">
			  </div>
			"""
		else:
			next_str = u"""<p style="text-align:center; font-size:160%;">  Comparison of tuned Gini and Entropy based Decision Tree Classifiers depicted in the form of confusion matrices. 
			A.) Decision Tree Classifier 
			B.) Decision Tree Classifier-Gini 
			C.) Decision Tree Classifier-Entropy 
			D.) """+ best_classifier_str + """ </p>
			<p style="text-align:center; font-size:100%;">  The original Decision Tree Classifier model was chosen as a base comparision and """+ best_classifier_str + """  was chosen since it showed the best average F1 Score </p>

			<div class="row">
			  <div class="column">
				  <p style="text-align:left; font-size:160%;">A.) Decision Tree Classifier <a href="../forDATA/""" + classifier_name + """_DecisionTreeClassifier.pickle" download> (Download) </a> </p>
				<img src=" """+ classifier_name +"""_DecisionTreeClassifier.png" alt="Snow" style="width:100%">
			  </div>
			"""

		file.write(next_str)

		if best_classifier_str == None :

			next_str = u"""
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

			next_str = u"""
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
		<p style="font-size:160%;">Comparison of statistics in the form of Accuracy, Precision, Recall and F1 Score calculated against the confusion matrices of respiration type for the classifiers</p>
		"""
		file.write(next_str)

		another_file = open(os.path.join(self.scratch, 'forHTML', 'html2folder', 'postStatistics.html'), u"r")
		all_str = another_file.read()
		another_file.close()

		file.write(all_str)

		next_str= u"""
		<p style="font-size:160%;"> Below is a tree created that displays a visual for how genomes were classified.</p>
		<p style="font-size:100%;"> READ: if __functional__role__ is absent (true) then move left otherwise if __functional__role__ is present (false) move right</p>

		<img src="NAMEmyTreeLATER.png" alt="Snow" style="width:100%">

		</body>
		</html>
		"""
		file.write(next_str)

		file.close()

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
		  <button class="tablinks" onclick="openTab(event, 'Overview')" id="defaultOpen">Main Page</button>
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
	def html_report_3(self):
		"""
		does: creates an .html file that makes the first report (second app).
		"""
		file = open(os.path.join(self.scratch, 'forSecHTML', 'html3.html'), u"w")

		html_string = u"""
		<!DOCTYPE html>
		<html>
		<head>
		<style>
		table, th, td {
			border: 1px solid black;
			border-collapse: collapse;
		}

		* {
			box-sizing: border-box;
		}

		.column {
			float: left;
			width: 50%;
			padding: 10px;
		}

		/* Clearfix (clear floats) */
		.row::after {
			content: "";
			clear: both;
			display: table;
		}
		</style>
		</head>
		<body>

		<h1 style="text-align:center;">Prediction Results</h1>

		<!-- <h2>Maybe we can add some more text here later?</h2> -->
		<!--<p>How to create side-by-side images with the CSS float property:</p> -->

		<p style="text-align:center; font-size:160%;">  Here is a simple table that shows the prediction for each sample and the probability of that prediction being correct </p>
		<p style="text-align:center; font-size:100%;">  (Remember you can always increase the probabilty of the prediction being correct by adding in more data in the build classifier app and then re-running this app. Good Luck!) </p>


		"""
		file.write(html_string)

		another_file = open(os.path.join(self.scratch, 'forSecHTML', 'html3folder', 'results.html'), u"r")
		all_str = another_file.read()
		another_file.close()

		file.write(all_str)

		next_str= u"""
		</body>
		</html>
		"""
		file.write(next_str)

		file.close()

		return "html3.html"

	def html_nodual(self, location):

		if location == "forHTML":
			file = open(os.path.join(self.scratch, 'forHTML', 'nodual.html'), u"w")
		else :
			file = open(os.path.join(self.scratch, 'forSecHTML', 'nodual.html'), u"w")

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
			  <button class="tablinks" onclick="openTab(event, 'Overview')" id="defaultOpen">Main Page</button>
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
		else :
			next_str = u"""
			  <div id="Overview" class="tabcontent">
				  <iframe src="html3.html" style="height:100vh; width:100%; border: hidden;" ></iframe>
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