import os
import re
import uuid
import time
import pickle
import operator
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
plt.switch_backend('agg')

#Add Parllelism
from concurrent.futures import ThreadPoolExecutor

#Classifier Models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier

#additional classifier methods
from sklearn.tree import export_graphviz
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

from KBaseReport.KBaseReportClient import KBaseReport
from DataFileUtil.DataFileUtilClient import DataFileUtil
from installed_clients.RAST_SDKClient import RAST_SDK
from installed_clients.kb_SetUtilitiesClient import kb_SetUtilities
from biokbase.workspace.client import Workspace as workspaceService


class kb_genomeclfUtils(object):
	def __init__(self, config):

		self.workspaceURL = config['workspaceURL']
		self.scratch = config['scratch']
		self.callback_url = config['callback_url']

		self.ctx = config['ctx']

		self.dfu = DataFileUtil(self.callback_url)
		self.rast = RAST_SDK(self.callback_url, service_ver='beta')
		self.kb_util = kb_SetUtilities(self.callback_url)
		self.ws_client = workspaceService(self.workspaceURL)

	def fullUpload(self, params, current_ws):
		"""
		workhorse function for upload_trainingset
		"""

		#create folder
		folder_name = "forUpload"
		os.makedirs(os.path.join(self.scratch, folder_name), exist_ok=True)

		#Testing Files
		#params["file_path"] = "/kb/module/data/RealData/GramDataEdit2Ref.xlsx"
		#params["file_path"] = "/kb/module/data/RealData/fake_2_refseq_simple.xlsx"
		#params["file_path"] = "/kb/module/data/RealData/SingleForJanaka.xlsx"
		#uploaded_df = pd.read_excel(params["file_path"], dtype=str)
		
		#Case Study Test
		#params["file_path"] = "/kb/module/data/RealData/respiration-benchmark.csv"
		#uploaded_df = pd.read_csv(params["file_path"], header=0, dtype=str)
		
		#True App
		uploaded_df = self.getUploadedFileAsDF(params["file_path"])
		(upload_table, classifier_training_set, missing_genomes, genome_label, number_of_genomes, number_of_classes) = self.createAndUseListsForTrainingSet(current_ws, params, uploaded_df)

		self.uploadHTMLContent(params['training_set_name'], params["file_path"], missing_genomes, genome_label, params['phenotype'], upload_table, number_of_genomes, number_of_classes)
		html_output_name = self.viewerHTMLContent(folder_name, status_view = True)
		
		return html_output_name, classifier_training_set

	def fullAnnotate(self, params, current_ws):
		"""
		workhorse function for rast_annotate_trainingset
		"""

		#create folder
		folder_name = "forAnnotate"
		os.makedirs(os.path.join(self.scratch, folder_name), exist_ok=True)

		training_set_name = params['training_set_name']

		training_set_object = self.ws_client.get_objects2({'objects' : [{'workspace':current_ws, 'name': training_set_name}]})["data"]
	
		phenotype = training_set_object[0]['data']["classification_type"]
		classes_sorted = training_set_object[0]['data']["classes"]

		training_set_object_data = training_set_object[0]['data']['classification_data']
		training_set_object_reference = training_set_object[0]['path'][0]

		_list_genome_name = []
		_list_genome_ref = []
		_list_phenotype = []
		_list_references = []
		_list_evidence_types = []

		for genome in training_set_object_data:
			_list_genome_name.append(genome["genome_name"])
			_list_genome_ref.append(genome["genome_ref"])
			_list_phenotype.append(genome["genome_classification"])
			_list_references.append(genome["references"])
			_list_evidence_types.append(genome["evidence_types"])


		genome_set_name = "RAST_"+training_set_name
		# self.RASTAnnotateGenome(current_ws, _list_genome_ref, genome_set_name)

		# RAST_genome_names = _list_genome_name
		# RAST_genome_references = _list_genome_ref
		#We know a head of time that all names are just old names with .RAST appended to them
		RAST_genome_names = [genome_set_name + "_" + genome_name  for genome_name in _list_genome_name]

		self.RASTAnnotateGenomeParallel(current_ws, _list_genome_ref, genome_set_name, _list_genome_name, RAST_genome_names)

		#Figure out new RAST references 
		RAST_genome_references = []
		for RAST_genome in RAST_genome_names:
			meta_data = self.ws_client.get_objects2({'objects' : [{'workspace':current_ws, 'name': RAST_genome}]})['data'][0]['info']
			genome_ref = str(meta_data[6]) + "/" + str(meta_data[0]) + "/" + str(meta_data[4])
			RAST_genome_references.append(genome_ref)

		# make the classifier_training_set (only need to make it from present genomes)
		classifier_training_set = {}
		for index, curr_genome_ref in enumerate(RAST_genome_references):
			classifier_training_set[curr_genome_ref] = { 	'genome_name': RAST_genome_names[index],
															'genome_ref': curr_genome_ref,
															'phenotype': _list_phenotype[index],
															'references': _list_references[index],
															'evidence_types': _list_evidence_types[index]
														}

		modified_params = {
		'training_set_name': params["annotated_trainingset_name"],
		'description': "We RAST Annotated " + training_set_name,
		'phenotype': phenotype
		}

		self.createTrainingSetObject(current_ws, modified_params, RAST_genome_names, RAST_genome_references, _list_phenotype, _list_references, _list_evidence_types, with_Rast=True)


		#Make Report Table
		report_table = pd.DataFrame.from_dict({	"Genome Name": _list_genome_name,
												"Annotated Genome Name": RAST_genome_names,
												})	

		self.annotateHTMLContent(params["annotated_trainingset_name"], genome_set_name, report_table)
		html_output_name = self.viewerHTMLContent(folder_name, status_view = True)
		
		return html_output_name, classifier_training_set


	def fullClassify(self, params, current_ws):
		"""
		workhorse function for build_classifier
		"""

		#create folder for images and data
		folder_name = "forBuild"
		os.makedirs(os.path.join(self.scratch, folder_name), exist_ok=True)
		os.makedirs(os.path.join(self.scratch, folder_name, "images"), exist_ok=True)
		os.makedirs(os.path.join(self.scratch, folder_name, "data"), exist_ok=True)

		#unload the training_set_object
		#class_enumeration : {'N': 0, 'P': 1}
		#uploaded_df is four columns: Genome Name | Genome Reference | Phenotype | Phenotype Enumeration
		(phenotype, class_enumeration, uploaded_df, training_set_object_reference) = self.unloadTrainingSet(current_ws, params['training_set_name'])

		#get functional_roles and make indicator matrix
		(indicator_matrix, master_role_list) = self.createIndicatorMatrix(uploaded_df, params["genome_attribute"])

		#split up training data
		splits = 10
		whole_X = indicator_matrix[master_role_list].values
		whole_Y = uploaded_df["Phenotype Enumeration"].values
		(list_train_index, list_test_index) = self.getKSplits(splits, whole_X, whole_Y)

		#figure out which classifier is getting made
		common_classifier_information = {
				'class_list_mapping' : class_enumeration,
				'attribute_data' : master_role_list,
				'attribute_type': params["genome_attribute"],
				'splits': splits,
				'list_train_index' : list_train_index,
				'list_test_index' : list_test_index,
				'whole_X' : whole_X,
				'whole_Y' : whole_Y,
				'training_set_ref' : training_set_object_reference,
				'description' : params["description"]
				}

		classifier_info_list = []

		dict_classification_report_dict = {}
		genome_classifier_object_names = []
		classifier_to_run = params["classifier_to_run"]
		if(classifier_to_run == "run_all"):

			list_classifier_types = ["k_nearest_neighbors", "gaussian_nb", "logistic_regression", "decision_tree_classifier", "support_vector_machine", "neural_network"]
			for classifier_type in list_classifier_types:
				current_classifier_object = {	"classifier_to_execute": self.getCurrentClassifierObject(classifier_type, params[classifier_type]),
												"classifier_type": classifier_type,
												"classifier_name": params["classifier_object_name"] + "_" + classifier_type
											}
				genome_classifier_object_names.append(current_classifier_object["classifier_name"])

				#this is a dictionary containing 'class 0': {'precision': 0.5, 'recall': 1.0, 'f1-score': 0.2}, 'accuracy'
				(classification_report_dict,individual_classifier_info) = self.executeClassifier(current_ws, common_classifier_information, current_classifier_object, folder_name)
				dict_classification_report_dict[classifier_type] = classification_report_dict
				classifier_info_list.append(individual_classifier_info)


			#handle Decision Tree Case
			(ddt_dict_classification_report_dict, dtt_classifier_info, top_20) = self.tuneDecisionTree(current_ws, common_classifier_information, params["classifier_object_name"], folder_name)
			dict_classification_report_dict["decision_tree_classifier_gini"] = ddt_dict_classification_report_dict["decision_tree_classifier_gini"]
			dict_classification_report_dict["decision_tree_classifier_entropy"] = ddt_dict_classification_report_dict["decision_tree_classifier_entropy"]
			classifier_info_list.append(dtt_classifier_info[0])
			classifier_info_list.append(dtt_classifier_info[1])


		else:
			current_classifier_object = {	"classifier_to_execute": self.getCurrentClassifierObject(classifier_to_run, params[classifier_to_run]),
											"classifier_type": classifier_to_run,
											"classifier_name": params["classifier_object_name"] + "_" + classifier_to_run
										}
			genome_classifier_object_names.append(current_classifier_object["classifier_name"])

			(classification_report_dict,individual_classifier_info)  = self.executeClassifier(current_ws, common_classifier_information, current_classifier_object, folder_name)
			dict_classification_report_dict[classifier_to_run] = classification_report_dict
			classifier_info_list.append(individual_classifier_info)

			if(classifier_to_run == "decision_tree_classifier"):
				(ddt_dict_classification_report_dict, dtt_classifier_info, top_20) = self.tuneDecisionTree(current_ws, common_classifier_information, params["classifier_object_name"], folder_name)
				dict_classification_report_dict["decision_tree_classifier_gini"] = ddt_dict_classification_report_dict["decision_tree_classifier_gini"]
				dict_classification_report_dict["decision_tree_classifier_entropy"] = ddt_dict_classification_report_dict["decision_tree_classifier_entropy"]
				classifier_info_list.append(dtt_classifier_info[0])
				classifier_info_list.append(dtt_classifier_info[1])


		#generate table from dict_classification_report_dict
		main_report_df_flag = False
		dtt_report_df_flag = False

		(main_report_df, dtt_report_df, best_classifier_type_nice, genome_dtt_classifier_object_names) = self.handleClassificationReports(dict_classification_report_dict, list(common_classifier_information["class_list_mapping"].keys()), params["classifier_object_name"] )
		if(len(main_report_df.keys()) > 1):
			main_report_df_flag = True
			self.buildMainHTMLContent(params['training_set_name'], main_report_df, genome_classifier_object_names, phenotype, best_classifier_type_nice)
		if(len(dtt_report_df.keys()) > 0):
			dtt_report_df_flag = True
			self.buildDTTHTMLContent(dtt_report_df, top_20, genome_dtt_classifier_object_names, best_classifier_type_nice)

		html_output_name = self.viewerHTMLContent(folder_name, main_report_view = main_report_df_flag, decision_tree_view = dtt_report_df_flag)

		return html_output_name, classifier_info_list

	def getCurrentClassifierObject(self, classifier_type, params):
		"""
		Takes user selected classifier_type (as string from dropdown) and
		returns corresponding sklearn classfifier with user selected parameters (or defaults)

		Parameter
		---------
		classifier_type : str
			"k_nearest_neighbors, gaussian_nb, etc."
		params : dict
			parameters for sklearn classifier
		"""

		if classifier_type == "k_nearest_neighbors":
			if(params == None):
				return KNeighborsClassifier()
			else:
				return KNeighborsClassifier(n_neighbors = params["n_neighbors"],
											weights = params["weights"],
											algorithm = params["algorithm"],
											leaf_size = params["leaf_size"],
											p = params["p"],
											metric = params["metric"]
											)

		elif classifier_type == "gaussian_nb":
			if(params == None):
				return GaussianNB()
			else:
				if(params["priors"] == "None"):
					return GaussianNB(priors=None)
				else:
					return GaussianNB(	priors=params["prior"]
										)

		elif classifier_type == "logistic_regression":
			if(params == None):
				return LogisticRegression(random_state=0)
			else:
				return LogisticRegression(	penalty = params["penalty"],
											dual = self.getBool(params["dual"]),
											tol = params["lr_tolerance"],
											C = params["lr_C"],
											fit_intercept = self.getBool(params["fit_intercept"]),
											intercept_scaling = params["intercept_scaling"],
											solver = params["lr_solver"],
											max_iter = params["lr_max_iter"],
											multi_class = params["multi_class"]
											)

		elif classifier_type == "decision_tree_classifier":
			if(params == None):
				return DecisionTreeClassifier(random_state=0)
			else:

				return DecisionTreeClassifier(	criterion = params["criterion"],
												splitter = params["splitter"],
												max_depth = params["max_depth"],
												min_samples_split = params["min_samples_split"],
												min_samples_leaf = params["min_samples_leaf"],
												min_weight_fraction_leaf = params["min_weight_fraction_leaf"],
												max_leaf_nodes = params["max_leaf_nodes"],
												min_impurity_decrease = params["min_impurity_decrease"]
												)

		elif classifier_type == "support_vector_machine":
			if(params == None):
				return svm.SVC(kernel = "linear",random_state=0)
			else:
				return svm.SVC(	C = params["svm_C"],
								kernel = params["kernel"],
								degree = params["degree"],
								gamma = params["gamma"],
								coef0 = params["coef0"],
								probability = self.getBool(params["probability"]),
								shrinking = self.getBool(params["shrinking"]),
								tol = params["svm_tolerance"],
								cache_size = params["cache_size"],
								max_iter = params["svm_max_iter"],
								decision_function_shape = params["decision_function_shape"]
								)

		elif classifier_type == "neural_network":
			if(params == None):
				return MLPClassifier(random_state=0)
			else:
				return MLPClassifier(	hidden_layer_sizes = (int(params["hidden_layer_sizes"]),),
										activation = params["activation"],
										solver = params["mlp_solver"],
										alpha = params["alpha"],
										batch_size = params["batch_size"],
										learning_rate = params["learning_rate"],
										learning_rate_init = params["learning_rate_init"],
										power_t = params["power_t"],
										max_iter = params["mlp_max_iter"],
										shuffle = self.getBool(params["shuffle"]),
										tol = params["mlp_tolerance"],
										momentum = params["momentum"],
										nesterovs_momentum = self.getBool(params["nesterovs_momentum"]),
										early_stopping = self.getBool(params["early_stopping"]),
										validation_fraction = params["validation_fraction"],
										beta_1 = params["beta_1"],
										beta_2 = params["beta_2"],
										epsilon = params["epsilon"]
										)

	def getBool(self, value):
		"""
		Converts string to boolean

		Parameter
		---------
		value : str
			"False", "True"
		"""

		if(value == "False"):
			return False
		else:
			return True

	def executeClassifier(self, current_ws, common_classifier_information, current_classifier_object, folder_name):
		"""
		Creates k=splits number of classifiers and then generates a confusion matrix that averages
		over the predicted results for all of the classifiers.

		Generates statistics for each classifier (saved in classification_report_dict)

		Saves each classifier object as a pickle file and then uploads that object to shock, saves the object's 
		shock handle into the KBASE Categorizer Object (https://narrative.kbase.us/#spec/type/KBaseClassifier.GenomeCategorizer)

		Calls function to create png of confusion matrix

		Saves information for callback of build_classifier in individual_classifier_info

		Parameter
		---------
		current_ws : str
			current_ws
		common_classifier_information : dict
			information that is common to the current classifier that is going to be built
		current_classifier_object: dict
			information that is specific to the current classifier that is going to be built
		folder_name:
			folder name gives the location to save classifier and images
		"""

		individual_classifier_info = {}
		matrix_size = len(common_classifier_information["class_list_mapping"])
		cnf_matrix_proportion = np.zeros(shape=(matrix_size, matrix_size))
		
		classifier = current_classifier_object["classifier_to_execute"]

		for c in range(common_classifier_information["splits"]):
			X_train = common_classifier_information["whole_X"][common_classifier_information["list_train_index"][c]]
			y_train = common_classifier_information["whole_Y"][common_classifier_information["list_train_index"][c]]
			X_test = common_classifier_information["whole_X"][common_classifier_information["list_test_index"][c]]
			y_test = common_classifier_information["whole_Y"][common_classifier_information["list_test_index"][c]]

			classifier.fit(X_train, y_train)
			y_pred = classifier.predict(X_test)

			cnf = confusion_matrix(y_test, y_pred, labels=list(common_classifier_information["class_list_mapping"].values()))
			cnf_f = cnf.astype('float') / cnf.sum(axis=1)[:, np.newaxis]
			for i in range(len(cnf)):
				for j in range(len(cnf)):
					cnf_matrix_proportion[i][j] += cnf_f[i][j]

		#get statistics for the last case made
		#diagonal entries of cm are the accuracies of each class
		target_names = list(common_classifier_information["class_list_mapping"].keys())
		classification_report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict = True)

		#save down classifier object in pickle format
		pickle_out = open(os.path.join(self.scratch, folder_name, "data", current_classifier_object["classifier_name"] + ".pickle"), "wb")
		main_clf = classifier.fit(common_classifier_information["whole_X"], common_classifier_information["whole_Y"])
		pickle.dump(main_clf, pickle_out, protocol = 2)
		pickle_out.close()

		shock_id, handle_id = self._upload_to_shock(os.path.join(self.scratch, folder_name, "data", current_classifier_object["classifier_name"] + ".pickle"))

		classifier_object = {
		'classifier_id' : '',
		'classifier_type' : current_classifier_object["classifier_type"],
		'classifier_name' : current_classifier_object["classifier_name"],
		'classifier_data' : '', #saved in shock
		'classifier_handle_ref' : handle_id,
		'classifier_description' : common_classifier_information["description"],
		'lib_name' : 'sklearn',
		'attribute_type' : common_classifier_information["attribute_type"],
		'number_of_attributes' : len(common_classifier_information["attribute_data"]), #size of master_role_list
		'attribute_data' : common_classifier_information["attribute_data"],
		'class_list_mapping' : common_classifier_information["class_list_mapping"],
		'number_of_genomes' : len(common_classifier_information["whole_Y"]),
		'training_set_ref' : common_classifier_information["training_set_ref"]
		}

	
		obj_save_ref = self.ws_client.save_objects({'workspace': current_ws,
													  'objects':[{
													  'type': 'KBaseClassifier.GenomeCategorizer',
													  'data': classifier_object,
													  'name': current_classifier_object["classifier_name"],  
													  'provenance': self.ctx['provenance']
													  }]
													})[0]
	 
		#information for call back
		individual_classifier_info = {	"classifier_name": current_classifier_object["classifier_name"],
										"classifier_ref": obj_save_ref,
										"accuracy": classification_report_dict["accuracy"]}

		cm = np.round(cnf_matrix_proportion/common_classifier_information["splits"]*100.0,1)
		title = "CM: " + current_classifier_object["classifier_type"]
		self.plot_confusion_matrix(cm, title, current_classifier_object["classifier_name"], list(common_classifier_information["class_list_mapping"].keys()), folder_name)

		return classification_report_dict, individual_classifier_info

	def handleClassificationReports(self, dict_classification_report_dict, target_names, classifier_object_name):
		"""
		This function takes all of the classification scores (Precision, Recall, F1-Score, Accuracy) for all of the classifier
		and places them into a DataFrame that can then be transformed into an html report to show statistics for the classifiers

		The Build Classifer App can create 2 views:
		1. Main Report (main_report_dict/main_report_df)
			a. Run All 
				In the Run All case the user has selected to build all of the default classifier
				and so the view reflects this choice by showing all of the classifier and their appropriate
				scores

				This will also keep track of the best classifier in terms of accuracy and pass this information
				to the Decision Tree Report

				(will also make Decision Tree Report, since decision_tree is one of the options in Run All)

			b. Single Selection
				In the Single Selection case the user has selects only a single option so only one column of 
				statistics being shown

				however in the even that the user selects the Decision Tree as the single selection, then there is
				NO main page made and only a Decision Tree Report made
	
		2. Decision Tree Report (dtt_report_dict/dtt_report_df)
			Show statistics for vanilla Decision Tree Classifier, Best Gini Decision Tree, Best Entropy Decision Tree

			optionally if Run All has been run it will also repeat the statistics in the Main Report for the best classifier for comparision

		Parameter
		---------
		dict_classification_report_dict : dict
			classification scores per classifier
			ex: dict_classification_report_dict["decision_tree_classifier"]["aerobic"]["precision"]

		target_names : str list
			list of phenotypes ["aerobic", "anerobic", etc.]
		classifier_object_name: str
			classifier_object_name
		"""

		main_report_dict = {}
		dtt_report_dict = {}
		best_classifier_type_nice = None
		genome_dtt_classifier_object_names = []
		#genome_dtt_classifier_object_names is a list of classifier_object_name + "_" + dtt_classifier_type

		metric_column =[]
		for target in target_names:
			metric_column.append(target)
			metric_column.append("Precision")
			metric_column.append("Recall")
			metric_column.append("F1-Score")

		metric_column.append("Accuracy")
		main_report_dict["Metrics"] = metric_column

		classifier_types_to_nice = {"k_nearest_neighbors": "K Nearest Neighbors", 
									"gaussian_nb": "Gaussian Naive Bayes", 
									"logistic_regression": "Logistic Regression",
									"decision_tree_classifier": "Decision Tree",
									"decision_tree_classifier_gini": "Decision Tree Gini",
									"decision_tree_classifier_entropy": "Decision Tree Entropy",
									"support_vector_machine": "Support Vector Machine", 
									"neural_network": "Neural Network"
									}

		#only making a single column
		if(len(dict_classification_report_dict.keys())==1):
			classifier_type = list(dict_classification_report_dict.keys())[0]
			classifier_type_column = []

			for target in target_names:
				classifier_type_column.append(None)
				classifier_type_column.append(dict_classification_report_dict[classifier_type][target]["precision"])
				classifier_type_column.append(dict_classification_report_dict[classifier_type][target]["recall"])
				classifier_type_column.append(dict_classification_report_dict[classifier_type][target]["f1-score"])

			#also add accuracy
			classifier_type_column.append(dict_classification_report_dict[classifier_type]["accuracy"])

			main_report_dict[classifier_types_to_nice[classifier_type]] = classifier_type_column

		elif(len(dict_classification_report_dict.keys())==3):
			#case where the keys are decision_tree_classifier, decision_tree_classifier_gini, and decision_tree_classifier_entropy
			
			dtt_report_dict["Metrics"] = metric_column
			dtt_classifier_types = ["decision_tree_classifier", "decision_tree_classifier_gini", "decision_tree_classifier_entropy"]
			for classifier_type in dtt_classifier_types:
				classifier_type_column = []

				for target in target_names:
					classifier_type_column.append(None)
					classifier_type_column.append(dict_classification_report_dict[classifier_type][target]["precision"])
					classifier_type_column.append(dict_classification_report_dict[classifier_type][target]["recall"])
					classifier_type_column.append(dict_classification_report_dict[classifier_type][target]["f1-score"])

				#also add accuracy
				classifier_type_column.append(dict_classification_report_dict[classifier_type]["accuracy"])

				dtt_report_dict[classifier_types_to_nice[classifier_type]] = classifier_type_column
				genome_dtt_classifier_object_names.append(classifier_object_name + "_" + classifier_type)


			#In this case there will be no main page and only a decision tree page

		else:
			#there is everything **and** we have to select a best classifier
			regular_classifier_types = ["k_nearest_neighbors", "gaussian_nb", "logistic_regression", "decision_tree_classifier", "support_vector_machine", "neural_network"]
			regular_classifier_type_to_accuracy = {}

			for classifier_type in regular_classifier_types:
				classifier_type_column = []

				for target in target_names:
					classifier_type_column.append(None)
					classifier_type_column.append(dict_classification_report_dict[classifier_type][target]["precision"])
					classifier_type_column.append(dict_classification_report_dict[classifier_type][target]["recall"])
					classifier_type_column.append(dict_classification_report_dict[classifier_type][target]["f1-score"])

				#also add accuracy
				regular_classifier_type_to_accuracy[classifier_type] = dict_classification_report_dict[classifier_type]["accuracy"]
				classifier_type_column.append(dict_classification_report_dict[classifier_type]["accuracy"])

				main_report_dict[classifier_types_to_nice[classifier_type]] = classifier_type_column

			best_classifier_type = max(regular_classifier_type_to_accuracy.items(), key=operator.itemgetter(1))[0]
			best_classifier_type_nice = classifier_types_to_nice[best_classifier_type]

			#handle decision_tree_classifier, decision_tree_classifier_gini, and decision_tree_classifier_entropy
			dtt_report_dict["Metrics"] = metric_column
			dtt_classifier_types = []
			dtt_classifier_types.append("decision_tree_classifier")
			if(best_classifier_type != "decision_tree_classifier"):
				dtt_classifier_types.append(best_classifier_type)
			
			dtt_classifier_types.append("decision_tree_classifier_gini")
			dtt_classifier_types.append("decision_tree_classifier_entropy")

			for classifier_type in dtt_classifier_types:
				classifier_type_column = []

				for target in target_names:
					classifier_type_column.append(None)
					classifier_type_column.append(dict_classification_report_dict[classifier_type][target]["precision"])
					classifier_type_column.append(dict_classification_report_dict[classifier_type][target]["recall"])
					classifier_type_column.append(dict_classification_report_dict[classifier_type][target]["f1-score"])

				#also add accuracy
				classifier_type_column.append(dict_classification_report_dict[classifier_type]["accuracy"])

				dtt_report_dict[classifier_types_to_nice[classifier_type]] = classifier_type_column
				genome_dtt_classifier_object_names.append(classifier_object_name + "_" + classifier_type)

		main_report_df = pd.DataFrame(main_report_dict)
		dtt_report_df = pd.DataFrame(dtt_report_dict)

		return(main_report_df, dtt_report_df, best_classifier_type_nice, genome_dtt_classifier_object_names)

	def tuneDecisionTree(self, current_ws, common_classifier_information, classifier_object_name, folder_name):
		"""
		This function attempt to tune the vanilla Decision Tree Classifier on 2 hyperparameters:
		tree_depth and criterion, to do so it does a grid search over a range of different tree depths
		[1, ... , 13] and both criterion (gini & entropy).

		It saves the average training/testing scores over each iteration and saves them to a png file

		The best Decision Tree produced for each criterion (wrt depth) is saved.

		Parameter
		---------
		current_ws : str
			current_ws
		common_classifier_information : dict
			information for classifier details
		classifier_object_name: str
			classifier_object_name to save
		folder_name: str
			location to save data
		"""

		range_start = 1
		iterations = 13
		ddt_dict_classification_report_dict = {}
		dtt_classifier_info = []

		#Gini Criterion
		training_avg = []
		training_std = []
		validation_avg = []
		validation_std = []

		for tree_depth in range(range_start, iterations):#notice here that tree depth must start at 1
			
			classifier = DecisionTreeClassifier(random_state=0, max_depth=tree_depth, criterion=u'gini')
			train_score = []
			validate_score = []

			for c in range(common_classifier_information["splits"]):
				X_train = common_classifier_information["whole_X"][common_classifier_information["list_train_index"][c]]
				y_train = common_classifier_information["whole_Y"][common_classifier_information["list_train_index"][c]]
				X_test = common_classifier_information["whole_X"][common_classifier_information["list_test_index"][c]]
				y_test = common_classifier_information["whole_Y"][common_classifier_information["list_test_index"][c]]

				classifier.fit(X_train, y_train)
				y_pred = classifier.predict(X_test)

				train_score.append(classifier.score(X_train, y_train))
				validate_score.append(classifier.score(X_test, y_test))

			training_avg.append(np.average(np.array(train_score)))
			training_std.append(np.std(train_score))
			validation_avg.append(np.average(validate_score))
			validation_std.append(np.std(validate_score))

		#Create Figure
		fig, ax = plt.subplots(figsize=(6, 6))
		plt.errorbar(np.arange(range_start,iterations), training_avg, yerr=training_std, fmt=u'o', label=u'Training set')
		plt.errorbar(np.arange(range_start,iterations), validation_avg, yerr=validation_std, fmt=u'o', label=u'Testing set')
		ax.set_ylim(ymin=0.0, ymax=1.1)
		ax.set_title("Gini Criterion")
		plt.xlabel('Tree Depth', fontsize=12)
		plt.ylabel('Accuracy', fontsize=12)
		plt.legend(loc='lower left')
		ax.grid(which='major', linestyle=':', linewidth='0.5', color='black')
		plt.savefig(os.path.join(self.scratch, folder_name, "images", "decision_tree_classifier_gini_depth.png"))
		
		best_gini_depth = np.argmax(validation_avg) + 1
		best_gini_accuracy_score = np.max(validation_avg)

		#Create Gini Genome Categorizer
		current_classifier_object = {	"classifier_to_execute": DecisionTreeClassifier(random_state=0, max_depth=best_gini_depth, criterion='gini'),
										"classifier_type": "decision_tree_classifier_gini",
										"classifier_name": classifier_object_name + "_" + "decision_tree_classifier_gini"
									}

		#this is a dictionary containing 'class 0': {'precision': 0.5, 'recall': 1.0, 'f1-score': 0.2}, 'accuracy'
		(classification_report_dict, individual_classifier_info) = self.executeClassifier(current_ws, common_classifier_information, current_classifier_object, folder_name)
		ddt_dict_classification_report_dict["decision_tree_classifier_gini"] = classification_report_dict
		dtt_classifier_info.append(individual_classifier_info)

		
		#Entropy Criterion
		training_avg = []
		training_std = []
		validation_avg = []
		validation_std = []

		for tree_depth in range(range_start, iterations):#notice here that tree depth must start at 1
			classifier = DecisionTreeClassifier(random_state=0, max_depth=tree_depth, criterion=u'entropy')
			train_score = []
			validate_score = []

			for c in range(common_classifier_information["splits"]):
				X_train = common_classifier_information["whole_X"][common_classifier_information["list_train_index"][c]]
				y_train = common_classifier_information["whole_Y"][common_classifier_information["list_train_index"][c]]
				X_test = common_classifier_information["whole_X"][common_classifier_information["list_test_index"][c]]
				y_test = common_classifier_information["whole_Y"][common_classifier_information["list_test_index"][c]]

				classifier.fit(X_train, y_train)
				y_pred = classifier.predict(X_test)

				train_score.append(classifier.score(X_train, y_train))
				validate_score.append(classifier.score(X_test, y_test))

			training_avg.append(np.average(np.array(train_score)))
			training_std.append(np.std(train_score))
			validation_avg.append(np.average(validate_score))
			validation_std.append(np.std(validate_score))

		fig, ax = plt.subplots(figsize=(6, 6))
		plt.errorbar(np.arange(range_start,iterations), training_avg, yerr=training_std, fmt=u'o', label=u'Training set')
		plt.errorbar(np.arange(range_start,iterations), validation_avg, yerr=validation_std, fmt=u'o', label=u'Testing set')
		ax.set_ylim(ymin=0.0, ymax=1.1)
		ax.set_title("Entropy Criterion")
		plt.xlabel('Tree Depth', fontsize=12)
		plt.ylabel('Accuracy', fontsize=12)
		plt.legend(loc='lower left')
		ax.grid(which='major', linestyle=':', linewidth='0.5', color='black')
		plt.savefig(os.path.join(self.scratch, folder_name, "images", "decision_tree_classifier_entropy_depth.png"))
		
		best_entropy_depth = np.argmax(validation_avg) + 1
		best_entropy_accuracy_score = np.max(validation_avg)

		#Create Gini Genome Categorizer
		current_classifier_object = {	"classifier_to_execute": DecisionTreeClassifier(random_state=0, max_depth=best_entropy_depth, criterion='entropy'),
										"classifier_type": "decision_tree_classifier_entropy",
										"classifier_name": classifier_object_name + "_" + "decision_tree_classifier_entropy" 
									}

		#this is a dictionary containing 'class 0': {'precision': 0.5, 'recall': 1.0, 'f1-score': 0.2}, 'accuracy'
		(classification_report_dict, individual_classifier_info) = self.executeClassifier(current_ws, common_classifier_information, current_classifier_object, folder_name)
		ddt_dict_classification_report_dict["decision_tree_classifier_entropy"] = classification_report_dict
		dtt_classifier_info.append(individual_classifier_info)

		if best_gini_accuracy_score > best_entropy_accuracy_score:
			top_20 = self.tree_code(DecisionTreeClassifier(random_state=0, max_depth=best_gini_depth, criterion='gini'), common_classifier_information)
		else:
			top_20 = self.tree_code(DecisionTreeClassifier(random_state=0, max_depth=best_entropy_depth, criterion='entropy'), common_classifier_information)
		
		return (ddt_dict_classification_report_dict, dtt_classifier_info, top_20)

	def tree_code(self, tree, common_classifier_information):
		"""
		Takes the Decision Tree and produces an human understandable tree png and find the list of 
		the top 20 most important attributed in identifying the splits in the tree

		Parameter
		---------
		tree : sklearn DecisionTreeClassifier object
			defined by calling function
		common_classifier_information : dict
			information for classifier details
		"""

		tree = tree.fit(common_classifier_information["whole_X"], common_classifier_information["whole_Y"])

		tree_contents = export_graphviz(tree, 
										out_file=None, 
										feature_names=common_classifier_information["attribute_data"],
										class_names=list(common_classifier_information["class_list_mapping"].keys()))

		initial_tree_contents = open(os.path.join(self.scratch, 'forBuild', 'initial_tree_contents.dot'), 'w')
		initial_tree_contents.write(tree_contents)
		initial_tree_contents.close()

		#start parsing the tree contents
		#The tree that is made by export_graphviz, is UGLY! we try to add color and more human interpretability to it
		tree_contents = tree_contents.replace('\\n', '')
		tree_contents = re.sub(r'<=[^;]+', r' (Absent)" , color="0.650 0.200 1.000"]', tree_contents)
		tree_contents = re.sub(r'[^"]*(class = )', r'', tree_contents)
		tree_contents = re.sub(r'shape=box] ;', r'shape=Mrecord] ; node [style=filled];', tree_contents)

		color_set = []
		for i in range(len(list(common_classifier_information["class_list_mapping"].keys()))):
			color_set.append('%.4f'%np.random.random() + " " + '%.4f'%np.random.random()+ " " + '0.900')

		for current_class, current_color in zip(list(common_classifier_information["class_list_mapping"].keys()), color_set):
			tree_contents = re.sub(r'("%s")' % current_class, r'\1, color = "%s"' % current_color, tree_contents)


		modified_tree_contents = open(os.path.join(self.scratch, 'forBuild', 'modified_tree_contents.dot'), "w")
		modified_tree_contents.write(tree_contents)
		modified_tree_contents.close()

		#take the tree dot file and turn it into an image
		os.system(u'dot -Tpng ' + os.path.join(self.scratch, 'forBuild', 'modified_tree_contents.dot') + ' >  '+ os.path.join(self.scratch, 'forBuild', 'images', "VisualDecisionTree.png"))
		
		#find the 20 "most important" functional roles
		genome_attribute_to_importance = pd.DataFrame({	common_classifier_information["attribute_type"]: common_classifier_information["attribute_data"], 
														'Importance': tree.feature_importances_})

		top_20 = genome_attribute_to_importance.sort_values("Importance", ascending = False).head(20)

		return top_20

	def _upload_to_shock(self, file_path):

		f2shock_out = self.dfu.file_to_shock({'file_path': file_path,
											  'make_handle': True})

		shock_id = f2shock_out.get('shock_id')
		handle_id = f2shock_out.get('handle').get('hid')

		return shock_id, handle_id

	def plot_confusion_matrix(self, cm, title, classifier_name, classes, folder_name):
		"""
		Creates a Confunsion Matrix with specs

		Parameter
		---------
		cm : np array
			defined by calling function
		title : str
			title
		classifier_name : str
			classifier_name
		classes : list
			["aerobic", "anerobic", etc.]
		folder_name: str
			location to place images
		"""

		fig, ax = plt.subplots(figsize=(4.5,4.5))

		sns_plot = sns.heatmap(cm, annot=True, ax = ax, cmap=u"Blues", fmt=".1f", square=True)
		ax = sns_plot

		ax.set_xlabel(u'Predicted Labels') 
		ax.set_ylabel(u'True Labels')

		ax.set_title(title)
		ax.xaxis.set_ticklabels(classes)
		ax.yaxis.set_ticklabels(classes)

		plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
		plt.tight_layout()

		fig = sns_plot.get_figure()
		fig.savefig(os.path.join(self.scratch, folder_name, "images", classifier_name +".png"), format=u'png')


	def unloadTrainingSet(self, current_ws, training_set_name):
		"""
		Take a Training Set Object and extracts "Genome Name", "Genome Reference", "Phenotype",
		and "Phenotype Enumeration" and places into a DataFrame

		class_enumeration will be something like: {'N': 0, 'P': 1} ie. phenotype --> number
	
		Parameter
		---------
		current_ws : str
			current_ws
		training_set_name : str
			training set to use for app
		"""
		training_set_object = self.ws_client.get_objects2({'objects' : [{'workspace':current_ws, 'name': training_set_name}]})["data"]
	
		phenotype = training_set_object[0]['data']["classification_type"]
		classes_sorted = training_set_object[0]['data']["classes"]

		class_enumeration = {} #{'N': 0, 'P': 1}
		for index, _class in enumerate(classes_sorted):
			class_enumeration[_class] = index

		training_set_object_data = training_set_object[0]['data']['classification_data']
		training_set_object_reference = training_set_object[0]['path'][0]

		_names = []
		_references = []
		_phenotypes = []
		_phenotype_enumeration = []

		for genome in training_set_object_data:
			_names.append(genome["genome_name"])
			_references.append(genome["genome_ref"])
			_phenotypes.append(genome["genome_classification"])
			
			_enumeration = class_enumeration[genome["genome_classification"]]
			_phenotype_enumeration.append(_enumeration)

		uploaded_df = pd.DataFrame(data={	"Genome Name": _names,
											"Genome Reference": _references,
											"Phenotype": _phenotypes,
											"Phenotype Enumeration": _phenotype_enumeration})

		return(phenotype, class_enumeration, uploaded_df, training_set_object_reference)

	def createIndicatorMatrix(self, uploaded_df, genome_attribute, master_role_list = None):
		"""
		Creates an indicator matrix of the following form

						Function 1	Function 2	Function 3	Function 4	Function 5 ...

		Genome Name 1		1			0			1			1			0
		Genome Name 2		1			1			1			1			0
		Genome Name 3		0			0			1			1			0
		Genome Name 4		1			1			1			1			0
		Genome Name 5		0			0			1			0			1

		The [Genome Name 1, Genome Name 2, etc.] list is defined in uploaded_df (Names or References doesn't matter)
		The [Function 1, Function 2, etc.] list is generated in this function...

		How?
		1. First loop over all genome objects to fine list of functional roles
		2. Place all functional roles found over all genomes into a set
		3. This set is sorted and made into list (alphabetically) and thus arranged to become Function 1, Function 2, etc.
			a.  This list of all functional roles present in the genomes that were uploaded during the training time
				will be known as master_role_list

		Then once we have master_role_list, we can simply populate the indicator matrix 

		Parameter
		---------
		uploaded_df : pd DataFrame
			refined user uploaded dataframe with genome names and references
		genome_attribute : str
			functional_roles, k-mers, protein sequence, etc.
		master_role_list: str list 
			only defined if function called from Predict Phenotype

			This is because if function is called from Predict Phenotype, we don't need to figure out what the
			list of all possible functional roles are, instead we just need to figure out which of the possible
			functional roles are also present in genomes we are predicting phenotypes for
		"""

		genome_references = uploaded_df["Genome Reference"].to_list()

		if "functional_roles" == genome_attribute:
			
			ref_to_role = {}

			if(master_role_list == None):
				master_role_set = set()

			for genome_ref in genome_references:
				genome_object_data = self.ws_client.get_objects2({'objects':[{'ref': genome_ref}]})['data'][0]['data']

				#figure out where functional roles are kept
				keys_location = genome_object_data.keys()
				if ("features" in keys_location):
					location_of_functional_roles = genome_object_data["features"]

				elif ("non_coding_features" in keys_location):
					location_of_functional_roles = genome_object_data["non_coding_features"]

				elif ("cdss" in keys_location):
					location_of_functional_roles = genome_object_data["cdss"]

				else:
					raise ValueError("The functional roles are not under features, non_coding_features, or cdss...")

				#either the functional roles are under function or functions (really stupid...)
				keys_function = location_of_functional_roles[0].keys()
				function_str = "function" if "function" in keys_function else "functions"

				list_functional_roles = []
				for functional_role in location_of_functional_roles:
					try:
						role_to_insert = functional_role[function_str]
						if " @ " in  role_to_insert:
							list_functional_roles.extend(role_to_insert.split(" @ "))
						elif " / " in role_to_insert:
							list_functional_roles.extend(role_to_insert.split(" / "))
						elif "; " in role_to_insert:
							list_functional_roles.extend(role_to_insert.split("; "))
						else:
							list_functional_roles.append(functional_role[function_str])
					except (RuntimeError, TypeError, ValueError, NameError):
						print("apparently some function list just don't have functions...")
						pass

				#create a mapping from genome_ref to all of its functional roles
				ref_to_role[genome_ref] = list_functional_roles

				if(master_role_list == None):
					#keep updateing a set of all functional roles seen so far
					master_role_set = master_role_set.union(set(list_functional_roles))

			if(master_role_list == None):
				#we are done looping over all genomes
				master_role_list = sorted(list(master_role_set))
				master_role_list.remove('')
			ref_to_indication = {}

			#make indicator rows for each 
			for genome_ref in genome_references:
				set_functional_roles = set(ref_to_role[genome_ref])
				matching_index = [i for i, role in enumerate(master_role_list) if role in set_functional_roles] 

				indicators = np.zeros(len(master_role_list))
				indicators[np.array(matching_index)] = 1

				ref_to_indication[genome_ref] = indicators.astype(int)


			indicator_matrix = pd.DataFrame.from_dict(data = ref_to_indication, orient='index', columns = master_role_list).reset_index().rename(columns={"index":"Genome Reference"})
		
			return (indicator_matrix, master_role_list)
		else:
			raise ValueError("Only classifiers based on functional roles have been impliemented please check back later")

	def getKSplits(self, splits, whole_X, whole_Y):
		"""
		Creates training and testing sets based on strified k=10 splits

		Parameter
		---------
		splits : int
			default = 10
		whole_X : np array
			indicator matrix with data for (genomes x functional roles)
		whole_Y: np array 
			Phenotype classes denoted as integers
		"""

		#This cross-validation object is a variation of KFold that returns stratified folds. 
		#The folds are made by preserving the percentage of samples for each class.
		list_train_index = []
		list_test_index = []
		skf = StratifiedKFold(n_splits=splits, random_state=0, shuffle=True)
		for train_idx, test_idx in skf.split(whole_X, whole_Y):
			list_train_index.append(train_idx)
			list_test_index.append(test_idx)

		return (list_train_index, list_test_index)

	def fullPredict(self, params, current_ws):
		"""
		workhorse function for predict_phenotype
		"""

		#create folder
		folder_name = "forPredict"
		os.makedirs(os.path.join(self.scratch, folder_name), exist_ok=True)

		#Load Information from Categorizer 
		categorizer_object = self.ws_client.get_objects2({'objects' : [{'workspace':current_ws, 'name':params['categorizer_name']}]})["data"]

		categorizer_handle_ref = categorizer_object[0]['data']['classifier_handle_ref']
		categorizer_file_path = self._download_shock(categorizer_handle_ref)

		master_role_list = categorizer_object[0]['data']['attribute_data']
		class_list_mapping = categorizer_object[0]['data']['class_list_mapping']
		genome_attribute = categorizer_object[0]['data']['attribute_type']

		current_categorizer = pickle.load(open(categorizer_file_path, "rb"))


		#Load Information from UploadedFile
		params["file_path"] = "/kb/module/data/RealData/GramDataEdit5.xlsx"
		uploaded_df = pd.read_excel(params["file_path"], dtype=str)
		#uploaded_df = self.getUploadedFileAsDF(params["file_path"], forPredict=True)
		(missing_genomes, genome_label, subset_uploaded_df, _in_workspace, _list_genome_name, _list_genome_ref) = self.createListsForPredictionSet(current_ws, params, uploaded_df)

		#get functional_roles and make indicator matrix
		(indicator_matrix, master_role_list) = self.createIndicatorMatrix(subset_uploaded_df, genome_attribute, master_role_list = master_role_list)
		whole_X = indicator_matrix[master_role_list].values
		#Make Predictions on uploaded file
		predictions_numerical = current_categorizer.predict(whole_X)

		#{'N': 0, 'P': 1} --> {0:'N', 1: 'P'}
		inv_map_class_list_mapping = {v: k for k, v in class_list_mapping.items()}
		predictions_phenotype = [] #map numerical to phenotype
		for numerical in predictions_numerical:
			predictions_phenotype.append(inv_map_class_list_mapping[numerical])
		prediction_probabilities = current_categorizer.predict_proba(whole_X)
		prediction_probabilities = np.max(prediction_probabilities, axis = 1) #predict_proba returns an genome by class_num probability matrix, you only want to select maximum

		#for callback structure and prediction set object
		_list_prediction_phenotype = []
		_list_prediction_probabilities = []

		#for use in report
		_prediction_phenotype = []
		_prediction_probabilities = []
		index = 0

		genome_iter = uploaded_df[genome_label].to_list()
		for genome in genome_iter:
			if(genome not in missing_genomes):
				_prediction_phenotype.append(predictions_phenotype[index])
				_prediction_probabilities.append(prediction_probabilities[index])
				_list_prediction_phenotype.append(predictions_phenotype[index])
				_list_prediction_probabilities.append(prediction_probabilities[index])	

				index +=1
			else:
				_prediction_phenotype.append("N/A")
				_prediction_probabilities.append("N/A")


		#construct prediction_set mapping
		prediction_set = {}
		for index, curr_genome_ref in enumerate(_list_genome_ref):
			prediction_set[curr_genome_ref] = { 'genome_name': _list_genome_name[index],
												'genome_ref': curr_genome_ref,
												'phenotype': _list_prediction_phenotype[index],
												'prediction_probabilities': _list_prediction_probabilities[index]
												}


		training_set_ref = categorizer_object[0]['data']['training_set_ref']
		phenotype = self.ws_client.get_objects2({'objects' : [{'ref': training_set_ref}]})["data"][0]['data']["classification_type"]

		predict_table = pd.DataFrame.from_dict({	genome_label: genome_iter,
													phenotype: _prediction_phenotype,
													"Verified to be in the Narrative": _in_workspace,
													"Probability": _prediction_probabilities
												})
		
		self.predictHTMLContent(params['categorizer_name'], phenotype, genome_attribute, params["file_path"], missing_genomes, genome_label, predict_table)
		html_output_name = self.viewerHTMLContent(folder_name, status_view = True)

		return html_output_name, prediction_set


	def generateHTMLReport(self, current_ws, folder_name, single_html_name, description, for_build_classifier = False):
		"""
		Creates KBaseReport from html file

		Parameter
		---------
		current_ws : str
			current_ws
		folder_name : str
			"forUpload" || "forAnnotate" || "forBuild" || "forPredict"
		single_html_name: str
			file name to display in KBASE Report (from viewerHTMLContent)
		description: str
			description
		for_build_classifier: bool
			True if function called from build_classifier
		"""

		report_shock_id = self.dfu.file_to_shock({	'file_path': os.path.join(self.scratch, folder_name),
													'pack': 'zip'})['shock_id']

		html_output = {
		'name' : single_html_name, #always viewer.html
		'shock_id': report_shock_id
		}

		report_params = {'message': '',
			 'workspace_name': current_ws,
			 'html_links': [html_output],
			 'direct_html_link_index': 0,
			 'html_window_height': 500,
			 'report_object_name': 'kb_classifier_report_' + str(uuid.uuid4())
			 }

		if for_build_classifier:
			output_file_links = []

			for file in os.listdir(os.path.join(self.scratch, 'forBuild', 'data')):
				output_file_links.append({	'path' : os.path.join(self.scratch, 'forBuild', 'data', file),
											'name' : file
											})

			report_params['file_links'] = output_file_links

		kbase_report_client = KBaseReport(self.callback_url, token=self.ctx['token'])
		report_output = kbase_report_client.create_extended_report(report_params)

		return report_output		

	### Helper Methods ###

	def _make_dir(self):
		dir_path = os.path.join(self.scratch, str(uuid.uuid4()))
		os.mkdir(dir_path)

		return dir_path

	def _download_shock(self, handle_id):
		dir_path = self._make_dir()

		file_path = self.dfu.shock_to_file({'handle_id': handle_id,
											'file_path': dir_path})['file_path']
		
		return file_path

	def getUploadedFileAsDF(self, file_path, forPredict=False):
		"""
		Reads xlsx/csv/tsv file from staging area and converts to pandas DataFrame

		Parameter
		---------
		file_path : str
			file selected from staging area by user
		forPredict : bool
			is True if function is being called from Predict Phenotype
		"""

		file_path = self.dfu.download_staging_file({'staging_file_subdir_path':file_path})['copy_file_path']

		if file_path.endswith('.xlsx'):
			uploaded_df = pd.read_excel(file_path, dtype=str)
		elif file_path.endswith('.csv'):
			uploaded_df = pd.read_csv(file_path, header=0, dtype=str)
		elif file_path.endswith('.tsv'):
			uploaded_df = pd.read_csv(file_path, sep='\t', header=0, dtype=str)
		else:
			raise ValueError('The following file type is not accepted, must be .xlsx, .csv, .tsv')

		self.checkValidFile(uploaded_df, forPredict)
		return uploaded_df

	def checkValidFile(self, uploaded_df, forPredict):
		"""
		Check that the uploaded_df has appropriate columns to use apps

		if function is called from Upload Training Set then required columns are:
			(Genome Name||Genome Reference) && Phenotype
		if function is called from Predict Phenotype then required columns are:
			Genome Name||Genome Reference

		Parameter
		---------
		uploaded_df : pd DataFrame
			current_ws
		forPredict : bool
			is True if function is being called from Predict Phenotype
		"""

		uploaded_df_columns = uploaded_df.columns

		if forPredict:
			if(("Genome Name" in uploaded_df_columns) or ("Genome Reference" in uploaded_df_columns)):
				pass
			else:
				raise ValueError('File must include Genome Name/Genome Reference')
		else:		
			if (("Ref Seq Ids" in uploaded_df_columns) or ("Genome Name" in uploaded_df_columns) or ("Genome Reference" in uploaded_df_columns)) and ("Phenotype" in uploaded_df_columns):
				pass
			else:
				raise ValueError('File must include Genome Name/Genome Reference and Phenotype as columns')

	def createAndUseListsForTrainingSet(self, current_ws, params, uploaded_df):
		"""
		This function does preprocessing that is necessary to create a Training Set Object
		1. Regardless of if the user only passes in the Genome Name or the Genome Reference, it will go and
			acquire both and pass them to be saved in the training set object
		2. If the user chooses to have their genomes RAST Annotated then we will annotate them
		3. Create a display report(report_table) that has information
			on whether their genomes in their uploaded file (uploaded_df) are present in their workspace (missing_genomes),
			which genomes were added to the training set, their corresponding phenotype, references, evidence
		4. Notice that the report_table and classifier_training_set are different, the report_table has information for both missing
			and present genomes, but the classifier_training_set only has information for present genomes.

		Parameter
		---------
		current_ws : str
			current_ws
		params : dict
			user specified parameters
		uploaded_df: pd DataFrame
			user uploaded file
		"""
		(genome_label, all_df_genome, missing_genomes) = self.findMissingGenomes(current_ws, params["workspace_id"], uploaded_df)

		uploaded_df_columns = uploaded_df.columns
		_references = []
		if("References" in uploaded_df_columns):
			has_references = True
			uploaded_df["References"].fillna("", inplace=True)
		else:
			has_references = False
		
		_evidence_types = []
		if("Evidence Types" in uploaded_df_columns):
			has_evidence_types = True
			uploaded_df["Evidence Types"].fillna("", inplace=True)
		else:
			has_evidence_types = False

		############################################################
		#subset dataframe to only include values that aren't missing
		filtered_uploaded_df = uploaded_df[~uploaded_df[genome_label].isin(missing_genomes)]

		if(genome_label == "Genome Reference"):
			#get name
			input_genome_references = filtered_uploaded_df["Genome Reference"].to_list()

			input_genome_names = []
			for genome_ref in input_genome_references:
				genome_name = str(self.ws_client.get_objects2({'objects' : [{'ref':genome_ref}]})['data'][0]['info'][1])
				input_genome_names.append(genome_name)

		else:
			#get references
			input_genome_references = []

			for genome in filtered_uploaded_df[genome_label]: #genome_label MUST be "Genome Name"
				meta_data = self.ws_client.get_objects2({'objects' : [{'workspace':current_ws, 'name': genome}]})['data'][0]['info']
				genome_ref = str(meta_data[6]) + "/" + str(meta_data[0]) + "/" + str(meta_data[4])
				input_genome_references.append(genome_ref)

			input_genome_names = filtered_uploaded_df["Genome Name"].to_list()

		"""
		At this point both
			input_genome_references
			input_genome_names

		will be populated regardelss of what genome_label is 
		"""

		if(params["annotate"]):
			
			#RAST Annotate the Genome
			output_genome_set_name = params['training_set_name'] + "_RAST"

			#We know a head of time that altl names are just old names with .RAST appended to them
			RAST_genome_names = [params['training_set_name'] + "_RAST_" + genome_name  for genome_name in input_genome_names]

			#output_genome_set_name = params['training_set_name'] + "_GenomeSET"
			self.RASTAnnotateGenomeParallel(current_ws, input_genome_references, output_genome_set_name, input_genome_names, RAST_genome_names)

			# genome_set_name = "RAST_"+training_set_name
			# # self.RASTAnnotateGenome(current_ws, _list_genome_ref, genome_set_name)

			# # RAST_genome_names = _list_genome_name
			# # RAST_genome_references = _list_genome_ref
			# #We know a head of time that all names are just old names with .RAST appended to them
			# RAST_genome_names = [genome_set_name + "_" + genome_name  for genome_name in _list_genome_name]

			# self.RASTAnnotateGenomeParallel(current_ws, _list_genome_ref, genome_set_name, _list_genome_name, RAST_genome_names)

			
			_list_genome_name = RAST_genome_names

			#Figure out new RAST references 
			RAST_genome_references = []
			for RAST_genome in RAST_genome_names:
				meta_data = self.ws_client.get_objects2({'objects' : [{'workspace':current_ws, 'name': RAST_genome}]})['data'][0]['info']
				genome_ref = str(meta_data[6]) + "/" + str(meta_data[0]) + "/" + str(meta_data[4])
				RAST_genome_references.append(genome_ref)
			_list_genome_ref = RAST_genome_references

		else:
			_list_genome_ref = input_genome_references

			#get genome_names
			genome_names = []
			for genome_ref in _list_genome_ref:
				name = str(self.ws_client.get_objects2({'objects' : [{'ref':genome_ref}]})['data'][0]['info'][1])
				genome_names.append(name)
			_list_genome_name = genome_names

		# get additional columns
		_list_phenotype = filtered_uploaded_df["Phenotype"].to_list()
		if(has_references):
			_list_references = filtered_uploaded_df["References"].str.split(";").to_list()
		else:
			_list_references = [[]]*filtered_uploaded_df.shape[0]

		if(has_evidence_types):
			_list_evidence_types = filtered_uploaded_df["Evidence Types"].str.split(";").to_list()
		else:
			_list_evidence_types = [[]]*filtered_uploaded_df.shape[0]

		# make the classifier_training_set (only need to make it from present genomes)
		classifier_training_set = {}
		for index, curr_genome_ref in enumerate(_list_genome_ref):
			classifier_training_set[curr_genome_ref] = { 	'genome_name': _list_genome_name[index],
															'genome_ref': curr_genome_ref,
															'phenotype': _list_phenotype[index],
															'references': _list_references[index],
															'evidence_types': _list_evidence_types[index]
														}


		#everything above this is only for non-missing genomes
		############################################################

		report_table = pd.DataFrame.from_dict({	genome_label: uploaded_df[genome_label],
												"Phenotype/Classification": uploaded_df["Phenotype"],
												})	

												#locations where genomes are present / not missing
		report_table["Verified to be in the Narrative"] = np.where(~uploaded_df[genome_label].isin(missing_genomes), "Yes", "No")
		report_table["Integrated into the Training Set"] = np.where(~uploaded_df[genome_label].isin(missing_genomes), "Yes", "No")

		if(has_references):
			report_table["References"] = uploaded_df["References"].str.split(";")
		if(has_evidence_types):
			report_table["Evidence Type(e.g; Respiration)"] = uploaded_df["Evidence Types"].str.split(";")

		(number_of_genomes, number_of_classes) = self.createTrainingSetObject(current_ws, params, _list_genome_name, _list_genome_ref, _list_phenotype, _list_references, _list_evidence_types)
		return (report_table, classifier_training_set, missing_genomes, genome_label, number_of_genomes, number_of_classes)

	def createTrainingSetObject(self, current_ws, params, _list_genome_name, _list_genome_ref, _list_phenotype, _list_references, _list_evidence_types, with_Rast = False):
		"""
		Creates a GenomeClassifierTrainingSet: 
		https://narrative.kbase.us/#spec/type/KBaseClassifier.GenomeClassifierTrainingSet

		Parameter
		---------
		current_ws : str
			current_ws
		params : dict
			user specified parameters
		_list_genome_name: str list
			list of genome names
		_list_genome_ref: str list
			list of genome references
		_list_phenotype: str list
			list of phenotypes
		_list_references: str list
			list of references
		_list_evidence_types: str list
			list of evidence types
		"""

		classification_data = []

		for index, curr_genome_ref in enumerate(_list_genome_ref):
			classification_data.append({ 	'genome_name': _list_genome_name[index],
											'genome_ref': curr_genome_ref,
											'genome_classification': _list_phenotype[index],
											'genome_id': "", #genome_id set to "" for now...
											'references': _list_references[index],
											'evidence_types': _list_evidence_types[index]
										})

		training_set_object = {
			'name': params['training_set_name'],
			'description': params['description'],
			'classification_type': params['phenotype'],
			'number_of_genomes': len(_list_genome_name),
			'number_of_classes': len(list(set(_list_phenotype))),
			'classes': sorted(list(set(_list_phenotype))),
			'classification_data': classification_data
			}

		if(with_Rast):
			training_set_object["annoated"] = True

		number_of_genomes = len(_list_genome_name)
		number_of_classes = len(list(set(_list_phenotype)))
		training_set_ref = self.ws_client.save_objects({'workspace': current_ws,
													  'objects':[{
																  'type': 'KBaseClassifier.GenomeClassifierTrainingSet',
																  'data': training_set_object,
																  'name': params['training_set_name'],  
																  'provenance': self.ctx['provenance']
																}]
													})[0]

		print("A Training Set Object named " + str(params['training_set_name']) + " with reference: " + str(training_set_ref) + " was just made.")
		
		return (number_of_genomes, number_of_classes)

	def checkUniqueColumn(self, uploaded_df, genome_label):

		if(uploaded_df[genome_label].is_unique):
			pass
		else:
			raise ValueError(str(genome_label) + " column is not unique")

	def genomes_to_ws(self, to_ws='', from_ws='19217', refseq_ids=[], verbose=False):
		"""
		Special Thanks to https://github.com/braceal

		Copies genomes specified in refseq_ids (GCF ids) from from_ws to to_ws
		and returns a dictionary of associated genome object ref ids e.g. 65797/3/1
		(ws/object-id/version).
		
		Will be GCF id itself if it does not exist
		
		return obj_refs: {'GCF_900128725.1': '36230/794/9', 'GCF_x001289725.1': 'GCF_900128725.1'}

		Parameter
		---------
		to_ws : str
			Workspace to copy objects to
		from_ws : str
			Workspace to copy objects from
		refseq_ids : list
			GCF ids of genomes to copy to to_ws
		verbose : bool
			If true, shows verbose output with progress updates
		"""
		# Default to current user workspace

		obj_refs = {curr_refseq_id: curr_refseq_id for curr_refseq_id in refseq_ids}
		total_add_refs = 0

		# Use set for O(1) query
		refseqs = set(refseq_ids)
		# Data batch parameters
		step = 10000 # How many genome objects to pull in each batch
		max_object_id = 0 # Keeps track of max object id seen so far
		prev_max_object_id = -1 # Defines stoping condition for while loop
		if verbose:
			batch = 0
		# If the previous iterations max object id is greater than or equal
		# to the most recent max_object_id, then all genomes have been pulled.
		while max_object_id > prev_max_object_id:
			prev_max_object_id = max_object_id
			# Get list of KBaseGenomes.Genome objects
			genomes = self.ws_client.list_objects({'ids':[from_ws],
									   'type': 'KBaseGenomes.Genome',
									   'includeMetadata':1,
									   'minObjectID':max_object_id,
									   'limit': step})
			if verbose:
				print(f'Batch {batch}')
				print(f'\t{len(genomes)} genomes pulled with max_object_id {max_object_id}')
				batch += 1
			# Check each received genome to see if it is in the user requested refseq_ids
			for genome in genomes:
				if genome[1] in refseqs:
					obj = self.ws_client.copy_object({'from':{'objid':genome[0],'wsid': from_ws},'to':{'wsid': to_ws,'name':genome[1]}})
					# Build list of KBase object references
					curr_obj_ref = f'{to_ws}/{obj[0]}/{obj[4]}'# (ws/object-id/version)
					obj_refs[genome[1]] = curr_obj_ref
					total_add_refs+=1
				# Early stopping optimization
				if total_add_refs == len(refseq_ids):
					return obj_refs
				# For pulling batches
				if genome[0] > max_object_id:
					max_object_id = genome[0]
		return obj_refs

	def findMissingGenomes(self, current_ws, workspace_id, uploaded_df):
		"""
		Finds missing genomes from user uploaded_df. Returns either a list of 
		Genome References or Genome Names that are missing from workspace but are 
		specified in uploaded_df. (Genome References are given preference over Genome Names)

		Parameter
		---------
		current_ws : str
			current_ws
		workspace_id: str
			ws_id ex: 69058 or 36230
		uploaded_df : pd DataFrame
			user uploaded dataframe
		"""

		if "Ref Seq Ids" in uploaded_df.columns:
			#in the event that the user passes in "Ref Seq Ids" you have to use those first
			self.checkUniqueColumn(uploaded_df, "Ref Seq Ids")

			#{'GCF_900128725.1': '36230/794/9', 'GCF_x001289725.1': 'GCF_x001289725.1'}
			# obj_refs = self.genomes_to_ws("36230", refseq_ids=uploaded_df["Ref Seq Ids"].to_list())
			obj_refs = self.genomes_to_ws(workspace_id, refseq_ids=uploaded_df["Ref Seq Ids"].to_list())
			#uploaded_df["Genome Reference"] = uploaded_df["Ref Seq Ids"].map(obj_refs)
			uploaded_df["Genome Name"] = uploaded_df["Ref Seq Ids"]

		all_genomes_workspace = self.ws_client.list_objects({'workspaces':[current_ws],'type':'KBaseGenomes.Genome'})

		#figure out if matching on Reference or Name and then find missing genomes
		if "Genome Reference" in uploaded_df.columns:
			genome_label = "Genome Reference"
			self.checkUniqueColumn(uploaded_df, genome_label)

			all_df_genome = uploaded_df[genome_label]
			all_refs2_workspace = [str(genome[6]) + "/" +str(genome[0])for genome in all_genomes_workspace]
			all_refs3_workspace = [str(genome[6]) + "/" +str(genome[0]) + "/" + str(genome[4]) for genome in all_genomes_workspace]
			
			missing_genomes = []
			for ref in all_df_genome:
				if((ref not in all_refs2_workspace) and (ref not in all_refs3_workspace)):
					missing_genomes.append(ref)

		else:
			genome_label = "Genome Name"
			self.checkUniqueColumn(uploaded_df, genome_label)

			all_df_genome = uploaded_df[genome_label]
			all_names_workspace = [str(genome[1]) for genome in all_genomes_workspace]
			missing_genomes = list(set(all_df_genome).difference(set(all_names_workspace)))

		"""
		genome_label is like the "key" that we are going to be consistently indexing by
		
		In this method if user passes in Ref Seq Ids ---> Genome Name
		"""

		return (genome_label, all_df_genome, missing_genomes)

	def RASTAnnotateGenome(self, current_ws, input_genomes, output_genome_set_name):
		"""
		Take a list of genome references and creates a RAST annotated genome_set
		based on (beta version of RAST SDK as of 8/2/2020)

		Call to rast_genomes_assemblies also adds the RAST annotated genomes 
		into the workspace.

		Parameter
		---------
		current_ws : str
			current_ws
		input_genomes : str list
			list of genome references ["12345/12/2", "12356/13/2", etc.]
		output_genome_set_name: str
			name of genome set that will be added to the workspaces: params['training_set_name'] + "_RAST"
		"""

		print("in RASTAnnotateGenome input_genomes are:")
		print(input_genomes)

		params_RAST =	{
		"input_text": ";".join(input_genomes),
		"output_workspace": current_ws,
		"output_GenomeSet_name" : output_genome_set_name
		}
		
		#we don't do anything with the output but you can if you want to
		print(params_RAST)
		output = self.rast.rast_genomes_assemblies(params_RAST)


		print("this is output from rast processing")
		print(output)
		print("this is output from rast processing keys")
		print(output.keys())

		if(output):
			pass
		else:
			print("output is: " + str(output))
			raise ValueError("for some reason unable to RAST Annotate Genome")

	def RASTAnnotateGenomeParallel(self, current_ws, input_genomes, output_genome_set_name, list_genome_names, RAST_genome_names):


		batch_size = 1 # batch_size (ie. number of genomes in a batch)
		#hello future coder! 

		genome_batches = [input_genomes[ind:ind+batch_size] for ind in range(0, len(input_genomes), batch_size)]
		#genome_batches something like: [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12]]
		
		lengths_genome_batches = [len(batch) for batch in genome_batches]
		#lengths_genome_batches something like: [10, 2]


		list_genome_set_names = [f'{output_genome_set_name}_{i}' for i in range(len(genome_batches))]
		#genome_set_name_1, genome_set_name_2, etc.

		broadcasted_prefixes = [item for item, count in zip(list_genome_set_names,lengths_genome_batches) for i in range(count)]
		#https://stackoverflow.com/questions/33382474/repeat-each-item-in-a-list-a-number-of-times-specified-in-another-list
		#broadcasted_prefixes is something like: [genome_set_name_1, genome_set_name_1,genome_set_name_1, ..., genome_set_name_2, genome_set_name_2]

		split_prefix_names = [prefix_set_name +"_"+ actual_name for prefix_set_name, actual_name in zip(broadcasted_prefixes, list_genome_names)]
		#split_prefix_names something like: [genome_set_name_1_actual_genome_name_1, genome_set_name_1_actual_genome_name_2, ...
		#									 genome_set_name_2_actual_genome_name_11, genome_set_name_2_actual_genome_name_12,]

		#list of dictionary for kwargs for 
		kwargs = [{'current_ws': current_ws,
		           'input_genomes': batch,
		           'output_genome_set_name': output_name} 
		           for batch, output_name in zip(genome_batches,list_genome_set_names)] 

		#see Alex for details
		_worker = lambda kwargs: self.RASTAnnotateGenome(**kwargs)

		with ThreadPoolExecutor() as executor:
		        for _ in executor.map(_worker, kwargs):
		             continue

		#############################################
		#Do some housekeeping in the workspace Yay!
		#############################################

		#1 Merge the genome sets into one genome set
		#get references of list_genome_set_names
		list_genome_set_refs = []

		for output_name in list_genome_set_names: #genome_label MUST be "Genome Name"
			meta_data = self.ws_client.get_objects2({'objects' : [{'workspace':current_ws, 'name': output_name}]})['data'][0]['info']
			genome_set_ref= str(meta_data[6]) + "/" + str(meta_data[0]) + "/" + str(meta_data[4])
			list_genome_set_refs.append(genome_set_ref)

		merge_params = {
		"desc": "merging " + ', '.join(list_genome_set_names),
		"input_refs":list_genome_set_refs,
		"output_name":output_genome_set_name,
		"workspace_name":current_ws
		}

		#do merge call
		self.kb_util.KButil_Merge_GenomeSets(merge_params)

		#2 Delete the said genome sets
		for genome_set_ref in list_genome_set_refs:
		    self.ws_client.delete_objects([{'workspace': current_ws, 'objid' : genome_set_ref.split("/")[1]}]) #get the objid ie. the 902 in 36230/902/1, 


		print("here is split_prefix_names")
		print(split_prefix_names)
		print("here is RAST_genome_names")
		print(RAST_genome_names)
		#3 rename all output genomes to a standard name
		for original_name, new_name in zip(split_prefix_names, RAST_genome_names):
			self.ws_client.rename_object({'obj':{"workspace":current_ws, "name":original_name}, 'new_name': new_name})

		time.sleep(10)

	def createListsForPredictionSet(self, current_ws, params, uploaded_df):
		"""
		Similar method to createAndUseListsForTrainingSet, however adapted for the Predict Phenotype app.
		Most differences are that the only required user input is Genome Name or Genome Reference so we don't 
		need to account for other details.

		Parameter
		---------
		current_ws : str
			current_ws
		params : dict
			user specified parameters
		uploaded_df: pd DataFrame
			user uploaded file
		"""

		(genome_label, all_df_genome, missing_genomes) = self.findMissingGenomes(current_ws, params['workspace_id'], uploaded_df)
		uploaded_df_columns = uploaded_df.columns

		############################################################
		#subset dataframe to only include values that aren't missing
		filtered_uploaded_df = uploaded_df[~uploaded_df[genome_label].isin(missing_genomes)]

		#get references
		if(genome_label == "Genome Reference"):
			input_genome_references = filtered_uploaded_df["Genome Reference"].to_list()

			input_genome_names = []
			for genome_ref in input_genome_references:
				genome_name = str(self.ws_client.get_objects2({'objects' : [{'ref':genome_ref}]})['data'][0]['info'][1])
				input_genome_names.append(genome_name)

		else:
			input_genome_references = []

			for genome in filtered_uploaded_df[genome_label]: #genome_label MUST be "Genome Name"
				meta_data = self.ws_client.get_objects2({'objects' : [{'workspace':current_ws, 'name': genome}]})['data'][0]['info']
				genome_ref = str(meta_data[6]) + "/" + str(meta_data[0]) + "/" + str(meta_data[4])
				input_genome_references.append(genome_ref)

			input_genome_names = filtered_uploaded_df["Genome Name"].to_list()

		"""
		At this point both
			input_genome_references
			input_genome_names

		will be populated regardelss of what genome_label is 
		"""

		if(params["annotate"]):
			
			#RAST Annotate the Genome
			output_genome_set_name = params['training_set_name'] + "_RAST"
			#output_genome_set_name = params['training_set_name'] + "_GenomeSET"
			self.RASTAnnotateGenome(current_ws, input_genome_references, output_genome_set_name)
			#We know a head of time that all names are just old names with .RAST appended to them
			RAST_genome_names = [params['training_set_name'] + "_RAST_" + genome_name  for genome_name in input_genome_names]






			_list_genome_name = RAST_genome_names

			#Figure out new RAST references 
			RAST_genome_references = []
			for RAST_genome in RAST_genome_names:
				meta_data = self.ws_client.get_objects2({'objects' : [{'workspace':current_ws, 'name': RAST_genome}]})['data'][0]['info']
				genome_ref = str(meta_data[6]) + "/" + str(meta_data[0]) + "/" + str(meta_data[4])
				RAST_genome_references.append(genome_ref)
			_list_genome_ref = RAST_genome_references

		else:
			_list_genome_ref = input_genome_references

			#get genome_names
			genome_names = []
			for genome_ref in _list_genome_ref:
				name = str(self.ws_client.get_objects2({'objects' : [{'ref':genome_ref}]})['data'][0]['info'][1])
				genome_names.append(name)
			_list_genome_name = genome_names


		#everything above this is only for non-missing genomes
		############################################################
		#locations where genomes are present / not missing
		_in_workspace = np.where(~uploaded_df[genome_label].isin(missing_genomes), "True", "False")


		subset_uploaded_df = pd.DataFrame(data={	"Genome Name": _list_genome_name,
													"Genome Reference": _list_genome_ref
												})

		return (missing_genomes, genome_label, subset_uploaded_df, _in_workspace, _list_genome_name, _list_genome_ref)


	def viewerHTMLContent(self, folder_name, status_view = False, main_report_view = False, decision_tree_view = False, ensemble_view = False):
		file = open(os.path.join(self.scratch, folder_name, 'viewer.html'), "w")

		header = u"""
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

			<div class="tab">
			""" 
		file.write(header)

		if(status_view):
			str_button = u"""
			<button class="tablinks" onclick="openTab(event, 'Status')" id="defaultOpen">Status</button>
			"""
			file.write(str_button)

		if(main_report_view):
			str_button = u"""
			<button class="tablinks" onclick="openTab(event, 'Overview')" id="defaultOpen">Overview</button>
			"""
			file.write(str_button)

		if(decision_tree_view):
			if(main_report_view):
				str_button = u"""
				<button class="tablinks" onclick="openTab(event, 'Decision Tree Tuning')">Decision Tree Tuning</button>
				"""
			else:
				str_button = u"""
				<button class="tablinks" onclick="openTab(event, 'Decision Tree Tuning')" id="defaultOpen">Decision Tree Tuning</button>
				"""
			file.write(str_button)
			
		if(ensemble_view):
			str_button = u"""
			<button class="tablinks" onclick="openTab(event, 'Ensemble Model')">Ensemble Model</button>
			"""
			file.write(str_button)

		remainder = u"""
		</div>
		<div id="Status" class="tabcontent">
		  <iframe src="status.html" style="height:100vh; width:100%; border: hidden;" ></iframe>
		</div>

		<div id="Overview" class="tabcontent">
		  <iframe src="main_report.html" style="height:100vh; width:100%; border: hidden;" ></iframe>
		</div>

		<div id="Decision Tree Tuning" class="tabcontent">
		  <iframe src="dtt_report.html" style="height:100vh; width:100%; border: hidden;" ></iframe>
		</div>

		<div id="Ensemble Model" class="tabcontent">
		  <iframe src="ensemble.html" style="height:100vh; width:100%; border: hidden;" ></iframe>
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
		document.getElementById("defaultOpen").click();
		</script>
		</body>
		</html>
		"""
		file.write(remainder)
		file.close()

		return "viewer.html"

	def uploadHTMLContent(self, training_set_name, selected_file_name, missing_genomes, genome_label, phenotype, upload_table, number_of_genomes, number_of_classes):
		
		file = open(os.path.join(self.scratch, 'forUpload', 'status.html'), "w")
		header = u"""
			<!DOCTYPE html>
			<html>
			<link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.21/css/jquery.dataTables.min.css">
			<body>

			<h2 style="text-align:center;"> Upload Training Set Data Summary </h2>
			<p>
			"""
		file.write(header)

		first_paragraph = u"""A Genome Classifier Training Set named """ + str(training_set_name) + """  \
							  with """ + str(number_of_genomes) + """ genomes and """ + str(number_of_classes) + """ unique classes was successfully created and added \
							  to the Narrative.</p><p>"""
		file.write(first_paragraph)


		second_paragraph = u"""The following table shows the information and the status for each genome that is integrated \
								into """ + str(training_set_name) + """.</p>"""

		file.write(second_paragraph)

		upload_table_html = upload_table.to_html(index=False, table_id="upload_table", justify='center')
		file.write(upload_table_html)

		scripts = u"""</body>

			<script type="text/javascript" src="https://code.jquery.com/jquery-3.5.1.js"></script>
			<script type="text/javascript" src="https://cdn.datatables.net/1.10.21/js/jquery.dataTables.min.js"></script>
			<script type="text/javascript">
			$(document).ready(function() {
				$('#upload_table').DataTable( {
					"scrollY":        "500px",
					"scrollCollapse": true,
					"paging":         false
				} );
			} );
			</script>
			</html>"""
		file.write(scripts)
		file.close()

	def annotateHTMLContent(self, annotated_trainingset_name, genome_set_name, report_table):
		file = open(os.path.join(self.scratch, 'forAnnotate', 'status.html'), "w")
		header = u"""
			<!DOCTYPE html>
			<html>
			<link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.21/css/jquery.dataTables.min.css">
			<body>

			<h2 style="text-align:center;"> Classifier Training Set Annotation Summary </h2>
			<p>
			"""
		file.write(header)

		first_paragraph = u"""The following genomes are annotated with RAST annotation algorithm. Additionally, \
							a genomeSet comprising of all RAST annotated genomes were created \
							named """ + str(genome_set_name) +  """.</p><p>"""
		file.write(first_paragraph)


		report_table_html = report_table.to_html(index=False, table_id="annotate_table", justify='center')
		file.write(report_table_html)

		scripts = u"""</body>

			<script type="text/javascript" src="https://code.jquery.com/jquery-3.5.1.js"></script>
			<script type="text/javascript" src="https://cdn.datatables.net/1.10.21/js/jquery.dataTables.min.js"></script>
			<script type="text/javascript">
			$(document).ready(function() {
				$('#annotate_table').DataTable( {
					"scrollY":        "500px",
					"scrollCollapse": true,
					"paging":         false
				} );
			} );
			</script>
			</html>"""
		file.write(scripts)
		file.close()

	def ulify(elements):
	    string = "<ul>\n"
	    for s in elements:
	        string += "<li>" + str(s) + "</li>\n"
	    string += "</ul>"
	    return string

	def buildMainHTMLContent(self, training_set_name, main_report_df, genome_classifier_object_names, phenotype, best_classifier_type_nice):

		folder_name = "forBuild"
		file = open(os.path.join(self.scratch, folder_name, 'main_report.html'), "w")
		header = u"""
			<!DOCTYPE html>
			<html>

			<style>
			figcaption{
			text-align: center;
			}
			* {
			  box-sizing: border-box;
			}
			.single{
			display: block;
			margin-left: auto;
			margin-right: auto;
			width: 40%;
			}
			.column {
			  float: left;
			  width: 50%;
			  padding: 5px;
			}

			/* Clearfix (clear floats) */
			.row::after {
			  content: "";
			  clear: both;
			  display: table;
			}
			</style>

			<link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.21/css/jquery.dataTables.min.css">
			<body>

			<h2 style="text-align:center;"> Construction of Genome Classifiers </h2>
			"""
		file.write(header)

		list_genome_classifier_object_names =  ulify(genome_classifier_object_names)
		file.write(list_genome_classifier_object_names)

		first_sentence = u"<p>Based on the training set """ + str(training_set_name) + """ .</p><p>"""
		file.write(first_sentence)

		first_paragraph = 	u"""Below is a confusion matrix (or matrices) which evaluates the performance of the selected classification algorithms \
							For each classification/phenotype class we compute the percentage of genomes with a true label (Y axis of the confusion matrix) \
							that get classified with a predicted label (X axis on the confusion matrix). Read more about <a href="https://en.wikipedia.org/wiki/Confusion_matrix"> Confusion Matrices </a>. Given the \
							magnitude of the number of classes relative to the overall size of the training set, creating only one test set would lead to \
							inconclusive results. Instead, we use <a href= "https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html"> K-Fold </a> (K=10) Cross Validation to ensure the quality of the model. Thus the below confusion \
							matrix (or matrices) represent the average percentages over all 10 folds.</p><p>
							"""
		file.write(first_paragraph)

		if(best_classifier_type_nice != None):
			sentence = u"""The best classification algorithm (highest average accuracy) was: """ + str(best_classifier_type_nice) +""".</p>"""
			file.write(sentence)


		#Do not produce confusion Matrix if more than 6 classes = 6*4 + 1
		if(main_report_df.shape[0] > 6*4+1):
			sentence = 	u"""<p color: #ff5050;> Sorry, we cannot display confusion matricies for classifiers with greater than 6 classes. \
						However statistics are still produced below.</p>"""
			file.write(sentence)

		#We are making a confusion Matrix with <6 classes
		else:
			#single classifier was choosen
			if(main_report_df.shape[1] == 2):
				images_str = u"""
							<div class="row">
								<figcaption>""" + genome_classifier_object_names[0] + """  <a href=" """+ os.path.join("data", genome_classifier_object_names[0] + ".pickle") + """ " download> (Download) </a> </figcaption>
								<img class="single" src=" """ + os.path.join("images", genome_classifier_object_names[0] +".png")+ """ " alt=" """+ genome_classifier_object_names[0]  +""" "  style="width:50%">
							</div>
				"""
				file.write(images_str)
			else:
				images_str = u"""
							<div class="row">
							  <div class="column">
								<figcaption>""" + genome_classifier_object_names[0] + """  <a href=" """+ os.path.join("data", genome_classifier_object_names[0] + ".pickle") + """ " download> (Download) </a> </figcaption>
								<img src=" """ + os.path.join("images", genome_classifier_object_names[0] +".png")+ """ " alt=" """+ genome_classifier_object_names[0]  +""" "  style="width:100%">
							  </div>
							  <div class="column">
								<figcaption>""" + genome_classifier_object_names[1] + """  <a href=" """+ os.path.join("data", genome_classifier_object_names[1] + ".pickle") + """ " download> (Download) </a> </figcaption>
								<img src=" """ + os.path.join("images", genome_classifier_object_names[1] +".png")+ """ " alt=" """+ genome_classifier_object_names[1]  +""" "  style="width:100%">
							  </div>
							</div>
							<div class="row">
							  <div class="column">
								<figcaption>""" + genome_classifier_object_names[2] + """  <a href=" """+ os.path.join("data", genome_classifier_object_names[2] + ".pickle") + """ " download> (Download) </a> </figcaption>
								<img src=" """ + os.path.join("images", genome_classifier_object_names[2] +".png")+ """ " alt=" """+ genome_classifier_object_names[2]  +""" "  style="width:100%">
							  </div>
							  <div class="column">
								<figcaption>""" + genome_classifier_object_names[3] + """  <a href=" """+ os.path.join("data", genome_classifier_object_names[3] + ".pickle") + """ " download> (Download) </a> </figcaption>
								<img src=" """ + os.path.join("images", genome_classifier_object_names[3] +".png")+ """ " alt=" """+ genome_classifier_object_names[3]  +""" "  style="width:100%">
							  </div>
							</div>
							<div class="row">
							  <div class="column">
								<figcaption>""" + genome_classifier_object_names[4] + """  <a href=" """+ os.path.join("data", genome_classifier_object_names[4] + ".pickle") + """ " download> (Download) </a> </figcaption>
								<img src=" """ + os.path.join("images", genome_classifier_object_names[4] +".png")+ """ " alt=" """+ genome_classifier_object_names[4]  +""" "  style="width:100%">
							  </div>
							  <div class="column">
								<figcaption>""" + genome_classifier_object_names[5] + """  <a href=" """+ os.path.join("data", genome_classifier_object_names[5] + ".pickle") + """ " download> (Download) </a> </figcaption>
								<img src=" """ + os.path.join("images", genome_classifier_object_names[5] +".png")+ """ " alt=" """+ genome_classifier_object_names[5]  +""" "  style="width:100%">
							  </div>
							</div>
				"""
			
				file.write(images_str)


		second_paragraph = 	u"""<p>For each classification algorithm, we also provide the Precision, Recall, and F1-Score for each """ + str(phenotype) + """ \
							class. More information about these metrics can be found <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support">
							here</a>.</p>
							"""
		file.write(second_paragraph)

		main_report_df.fillna('', inplace=True)
		main_report_html = main_report_df.to_html(index=False, table_id="main_report_table", justify='center')
		file.write(main_report_html)

		scripts = u"""</body>

			<script type="text/javascript" src="https://code.jquery.com/jquery-3.5.1.js"></script>
			<script type="text/javascript" src="https://cdn.datatables.net/1.10.21/js/jquery.dataTables.min.js"></script>

			<script type="text/javascript">
			$(document).ready(function() {
				$('#main_report_table').DataTable( {
					"ordering": false,
					scrollCollapse: true,
					paging:         false
				} );
			} );
			</script>
			</html>"""
		file.write(scripts)
		file.close()

	def buildDTTHTMLContent(self, dtt_report_df, top_20, genome_dtt_classifier_object_names, best_classifier_type_nice):

		folder_name = "forBuild"
		file = open(os.path.join(self.scratch, folder_name, 'dtt_report.html'), "w")
		header = u"""
			<!DOCTYPE html>
			<html>

			<style>
			figcaption{
			text-align: center;
			}
			* {
			  box-sizing: border-box;
			}
			.single{
			display: block;
			margin-left: auto;
			margin-right: auto;
			width: 40%;
			}
			.column {
			  float: left;
			  width: 50%;
			  padding: 5px;
			}

			/* Clearfix (clear floats) */
			.row::after {
			  content: "";
			  clear: both;
			  display: table;
			}
			</style>

			<link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.21/css/jquery.dataTables.min.css">
			<body>

			<h2 style="text-align:center;"> Decision Tree Tuning </h2>
			<p>
			"""
		file.write(header)

		first_paragraph = 	u"""Since the functional roles are categorical, we further fine tune the Decision Tree \
							classification Algorithm to seek better metrics. We tune the Decision Tree based on two \
							hyperparameters: Tree Depth and Criterion (quality of a split). The two criterion are "gini" which uses \
							the Gini impurity score and "entropy" which uses information gain score.</p>
							"""
		file.write(first_paragraph)


		#Show the graph for Tree Depth Accuracy
		tuning_str =u"""
		<div class="row">
		  <div class="column">
			<figcaption> Decision Tree Gini Criterion </figcaption>
			<img src=" """ + os.path.join("images", "decision_tree_classifier_gini_depth.png")+ """ " alt="decision_tree_classifier_gini_depth"  style="width:100%">
		  </div>
		  <div class="column">
			<figcaption> Decision Tree Entropy Criterion</figcaption>
			<img src=" """ + os.path.join("images", "decision_tree_classifier_entropy_depth.png")+ """ " alt="decision_tree_classifier_entropy_depth"  style="width:100%">
		  </div>
		</div>
		"""
		file.write(tuning_str)

		if((best_classifier_type_nice != "Decision Tree") and (best_classifier_type_nice != None)):
			sentence = 	u"""<p> We also include the confusion matrix and metrics for """ + best_classifier_type_nice + """ (the best classifier determined highest average accuracy)</p>"""
			file.write(sentence)
		
			tuning_str =u"""
			<div class="row">
			  <div class="column">
				<figcaption>""" + genome_dtt_classifier_object_names[0] + """  <a href=" """+ os.path.join("data", genome_dtt_classifier_object_names[0] + ".pickle") + """ " download> (Download) </a> </figcaption>
				<img src=" """ + os.path.join("images", genome_dtt_classifier_object_names[0] +".png")+ """ " alt=" """+ genome_dtt_classifier_object_names[0]  +""" "  style="width:100%">
			  </div>
			  <div class="column">
				<figcaption>""" + genome_dtt_classifier_object_names[1] + """  <a href=" """+ os.path.join("data", genome_dtt_classifier_object_names[1] + ".pickle") + """ " download> (Download) </a> </figcaption>
				<img src=" """ + os.path.join("images", genome_dtt_classifier_object_names[1] +".png")+ """ " alt=" """+ genome_dtt_classifier_object_names[1]  +""" "  style="width:100%">
			  </div>
			</div>
			"""
			file.write(tuning_str)

		else:	
			tuning_str = u"""
						<div class="row">
							<figcaption>""" + genome_dtt_classifier_object_names[0] + """  <a href=" """+ os.path.join("data", genome_dtt_classifier_object_names[0] + ".pickle") + """ " download> (Download) </a> </figcaption>
							<img class="single" src=" """ + os.path.join("images", genome_dtt_classifier_object_names[0] +".png")+ """ " alt=" """+ genome_dtt_classifier_object_names[0]  +""" "  style="width:50%">
						</div>
			"""
			file.write(tuning_str)
		
		tuning_str =u"""
		<div class="row">
		  <div class="column">
			<figcaption>""" + genome_dtt_classifier_object_names[-1 -1] + """  <a href=" """+ os.path.join("data", genome_dtt_classifier_object_names[-1 -1] + ".pickle") + """ " download> (Download) </a> </figcaption>
			<img src=" """ + os.path.join("images", genome_dtt_classifier_object_names[-1 -1] +".png")+ """ " alt=" """+ genome_dtt_classifier_object_names[-1 -1]  +""" "  style="width:100%">
		  </div>
		  <div class="column">
			<figcaption>""" + genome_dtt_classifier_object_names[-1] + """  <a href=" """+ os.path.join("data", genome_dtt_classifier_object_names[-1] + ".pickle") + """ " download> (Download) </a> </figcaption>
			<img src=" """ + os.path.join("images", genome_dtt_classifier_object_names[-1] +".png")+ """ " alt=" """+ genome_dtt_classifier_object_names[-1]  +""" "  style="width:100%">
		  </div>
		</div>
		"""
		file.write(tuning_str)

		sentence = u"""<p>Below is a visual reprsentation of the Decision Tree with the highest accuracy. Each node represents\
						a decision (True or False) that the model predicts during the classification process, if the \
						functional role is absent the classifier moves left, and if it is present it moves right. Leaf\
						nodes represent final classifications.</p>"""
		file.write(sentence)

		dtt_report_df.fillna('', inplace=True)
		dtt_report_html = dtt_report_df.to_html(index=False, table_id="dtt_report_table", justify='center')
		file.write(dtt_report_html)

		tree_image = u"""
					<br>
					<div class="row">
						<figcaption> Decision Tree on Functional Roles </figcaption>
						<img src=" """ + os.path.join("images", "VisualDecisionTree.png")+ """ " alt="VisualDecisionTree"  style="width:100%">
					</div>
		"""
		file.write(tree_image)


		sentence = u"""<p>Below is the <a href="https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier.feature_importances_">weighting scheme</a> \
						that the Decision Tree with the highest accuracy places on each of the functional roles</p>"""
		file.write(sentence)

		top_20_html = top_20.to_html(index=False, table_id="top_20_table", justify='center')
		file.write(top_20_html)

		scripts = u"""</body>

			<script type="text/javascript" src="https://code.jquery.com/jquery-3.5.1.js"></script>
			<script type="text/javascript" src="https://cdn.datatables.net/1.10.21/js/jquery.dataTables.min.js"></script>
			<script type="text/javascript">
			$(document).ready(function() {
				$('#dtt_report_table').DataTable( {
					"ordering": false,
					scrollCollapse: true,
					paging:         false
				} );
			} );
			</script>
			<script type="text/javascript">
			$(document).ready(function() {
				$('#top_20_table').DataTable( {
					"scrollY":        "500px",
					"scrollCollapse": true,
					"paging":         false
				} );
			} );
			</script>
			</html>"""
		file.write(scripts)
		file.close()

	def predictHTMLContent(self, categorizer_name, phenotype, selection_attribute, selected_file_name, missing_genomes, genome_label, predict_table):		
		file = open(os.path.join(self.scratch, 'forPredict', 'status.html'), u"w")
		header = u"""
			<!DOCTYPE html>
			<html>
			<link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.21/css/jquery.dataTables.min.css">
			<body>

			<h2 style="text-align:center;"> Phenotype Prediction based on Genome Classifiers </h2>
			<p>
			"""
		file.write(header)

		first_paragraph = u"""The Genome Categorizer """ + str(categorizer_name) \
							+ """ has been used to make predictions for  """ + str(phenotype)+ """ based on \
							""" + str(selection_attribute) + """. </p><p>"""
		file.write(first_paragraph)

		second_paragraph = u"""The following table summarizes the predictions in terms of <a href = "https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.predict_proba">probabilities</a>.</p>"""
		file.write(second_paragraph)

		predict_table_html = predict_table.to_html(index=False, table_id="predict_table", justify='center')
		file.write(predict_table_html)

		scripts = u"""</body>

			<script type="text/javascript" src="https://code.jquery.com/jquery-3.5.1.js"></script>
			<script type="text/javascript" src="https://cdn.datatables.net/1.10.21/js/jquery.dataTables.min.js"></script>
			<script type="text/javascript">
			$(document).ready(function() {
				$('#predict_table').DataTable( {
					"scrollY":        "500px",
					"scrollCollapse": true,
					"paging":         false
				} );
			} );
			</script>
			</html>"""
		file.write(scripts)
		file.close()



