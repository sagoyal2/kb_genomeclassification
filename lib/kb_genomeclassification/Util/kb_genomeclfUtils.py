import os
import re
import uuid
import pickle
import operator
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
plt.switch_backend('agg')

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

	#return html_output_name, classifier_training_set_mapping
	def fullUpload(self, params, current_ws):
		#create folder
		folder_name = "forUpload"
		os.makedirs(os.path.join(self.scratch, folder_name), exist_ok=True)

		#params["file_path"] = "/kb/module/data/RealData/GramDataEdit5.xlsx"
		#params["file_path"] = "/kb/module/data/RealData/full_genomeid_classification.xlsx"
		uploaded_df = self.getUploadedFileAsDF(params["file_path"])
		(upload_table, classifier_training_set, missing_genomes, genome_label) = self.createAndUseListsForTrainingSet(current_ws, params, uploaded_df)

		self.uploadHTMLContent(params['training_set_name'], params["file_path"], missing_genomes, genome_label, params['phenotype'], upload_table)
		html_output_name = self.viewerHTMLContent(folder_name, status_view = True)
		
		return html_output_name, classifier_training_set

	#return html_output_name, classifier_info_list, weight_list
	def fullClassify(self, params, current_ws):

		#double check that Ensemble is only called as specifid in the document
		# if((params["ensemble_model"]!=None) and (classifier_to_run != "run_all")):
		# 	raise ValueError("Ensemble Model will only be generated if Run All Classifiers is selected")


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


			#handle case for ensemble
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
			self.buildMainHTMLContent(main_report_df, genome_classifier_object_names, phenotype, best_classifier_type_nice)
		if(len(dtt_report_df.keys()) > 0):
			dtt_report_df_flag = True
			self.buildDTTHTMLContent(dtt_report_df, top_20, genome_dtt_classifier_object_names, best_classifier_type_nice)

		html_output_name = self.viewerHTMLContent(folder_name, main_report_view = main_report_df_flag, decision_tree_view = dtt_report_df_flag)

		return html_output_name, classifier_info_list

	def getCurrentClassifierObject(self, classifier_type, params):
		
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
		if(value == "False"):
			return False
		else:
			return True

	def executeClassifier(self, current_ws, common_classifier_information, current_classifier_object, folder_name):
		
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

		iterations = 13
		ddt_dict_classification_report_dict = {}
		dtt_classifier_info = []

		#Gini Criterion
		training_avg = []
		training_std = []
		validation_avg = []
		validation_std = []

		for tree_depth in range(1, iterations):#notice here that tree depth must start at 1
			
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
		plt.errorbar(np.arange(1,iterations), training_avg, yerr=training_std, fmt=u'o', label=u'Training set')
		plt.errorbar(np.arange(1,iterations), validation_avg, yerr=validation_std, fmt=u'o', label=u'Testing set')
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

		for tree_depth in range(1, iterations):#notice here that tree depth must start at 1
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
		plt.errorbar(np.arange(1,iterations), training_avg, yerr=training_std, fmt=u'o', label=u'Training set')
		plt.errorbar(np.arange(1,iterations), validation_avg, yerr=validation_std, fmt=u'o', label=u'Testing set')
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
			top_20 = self.tree_code(DecisionTreeClassifier(random_state=0, max_depth=best_gini_depth, criterion=u'gini'), common_classifier_information)
		else:
			top_20 = self.tree_code(DecisionTreeClassifier(random_state=0, max_depth=best_entropy_depth, criterion=u'entropy'), common_classifier_information)

		return (ddt_dict_classification_report_dict, dtt_classifier_info, top_20)

	def tree_code(self, tree, common_classifier_information):

		tree = tree.fit(common_classifier_information["whole_X"], common_classifier_information["whole_Y"])

		tree_contents = export_graphviz(tree, 
										out_file=None, 
										feature_names=common_classifier_information["attribute_data"],
										class_names=list(common_classifier_information["class_list_mapping"].keys()))

		initial_tree_contents = open(os.path.join(self.scratch, 'forBuild', 'initial_tree_contents.dot'), 'w')
		initial_tree_contents.write(tree_contents)
		initial_tree_contents.close()

		#start parsing the tree contents
		tree_contents = tree_contents.replace('\\n', '')
		tree_contents = re.sub(r'(\w\s\[label="[\w\s.,:\'\/()-]+)<=([\w\s.\[\]=,]+)("] ;)', r'\1 (Absent)" , color="0.650 0.200 1.000"] ;', tree_contents)
		tree_contents = re.sub(r'(\w\s\[label=")(.+?class\s=\s)', r'\1', tree_contents)
		tree_contents = re.sub(r'shape=box] ;', r'shape=Mrecord] ; node [style=filled];', tree_contents)

		color_set = []
		for i in range(len(list(common_classifier_information["class_list_mapping"].keys()))):
			color_set.append('%.4f'%np.random.random() + " " + '%.4f'%np.random.random()+ " " + '0.900')

		for current_class, current_color in zip(list(common_classifier_information["class_list_mapping"].keys()), color_set):
			tree_contents = re.sub(r'(\w\s\[label="%s")' % current_class, r'\1, color = "%s"' % current_color, tree_contents)


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

		training_set_object = self.ws_client.get_objects2({'objects' : [{'workspace':current_ws, 'name': training_set_name}]})["data"]
	
		phenotype = training_set_object[0]['data']["classification_type"]
		classes_sorted = training_set_object[0]['data']["classes"]

		class_enumeration = {}
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

		#This cross-validation object is a variation of KFold that returns stratified folds. The folds are made by preserving the percentage of samples for each class.
		list_train_index = []
		list_test_index = []
		skf = StratifiedKFold(n_splits=splits, random_state=0, shuffle=True)
		for train_idx, test_idx in skf.split(whole_X, whole_Y):
			list_train_index.append(train_idx)
			list_test_index.append(test_idx)

		return (list_train_index, list_test_index)

	def fullPredict(self, params, current_ws):
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
		#params["file_path"] = "/kb/module/data/RealData/GramDataEdit5.xlsx"
		uploaded_df = self.getUploadedFileAsDF(params["file_path"], forPredict=True)
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
													"In Workspace": _in_workspace,
													phenotype: _prediction_phenotype,
													"Probability": _prediction_probabilities
											 	})
		
		self.predictHTMLContent(params['categorizer_name'], phenotype, genome_attribute, params["file_path"], missing_genomes, genome_label, predict_table)
		html_output_name = self.viewerHTMLContent(folder_name, status_view = True)

		return html_output_name, prediction_set


	def generateHTMLReport(self, current_ws, folder_name, single_html_name, description, for_build_classifier = False):

		#folder_name = forUpload forBuild forPredict

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
		
		uploaded_df_columns = uploaded_df.columns

		if forPredict:
			if(("Genome Name" in uploaded_df_columns) or ("Genome Reference" in uploaded_df_columns)):
				pass
			else:
				raise ValueError('File must include Genome Name/Genome Reference')
		else:		
			if (("Genome Name" in uploaded_df_columns) or ("Genome Reference" in uploaded_df_columns)) and ("Phenotype" in uploaded_df_columns):
				pass
			else:
				raise ValueError('File must include Genome Name/Genome Reference and Phenotype as columns')

	def createAndUseListsForTrainingSet(self, current_ws, params, uploaded_df):
		
		(genome_label, all_df_genome, missing_genomes) = self.findMissingGenomes(current_ws, uploaded_df)

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

		if(params["annotate"]):
			
			#RAST Annotate the Genome
			output_genome_set_name = params['training_set_name'] + "_RAST_Genome_SET"
			self.RASTAnnotateGenome(current_ws, input_genome_references, input_genome_names, output_genome_set_name)

			#We know a head of time that all names are just old names with .RAST appended to them
			RAST_genome_names = [genome_name + ".RAST" for genome_name in input_genome_names]
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
												"Phenotype": uploaded_df["Phenotype"],
												})	

												#locations where genomes are present / not missing
		report_table["In Workspace"] = np.where(~uploaded_df[genome_label].isin(missing_genomes), "True", "False")
		report_table["In Training Set"] = np.where(~uploaded_df[genome_label].isin(missing_genomes), "True", "False")

		if(has_references):
			report_table["References"] = uploaded_df["References"].str.split(";")
		if(has_evidence_types):
			report_table["Evidence Types"] = uploaded_df["Evidence Types"].str.split(";")

		self.createTrainingSetObject(current_ws, params, _list_genome_name, _list_genome_ref, _list_phenotype, _list_references, _list_evidence_types)
		return (report_table, classifier_training_set, missing_genomes, genome_label)

	def createTrainingSetObject(self, current_ws, params, _list_genome_name, _list_genome_ref, _list_phenotype, _list_references, _list_evidence_types):

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

		training_set_ref = self.ws_client.save_objects({'workspace': current_ws,
													  'objects':[{
																  'type': 'KBaseClassifier.GenomeClassifierTrainingSet',
																  'data': training_set_object,
																  'name': params['training_set_name'],  
																  'provenance': self.ctx['provenance']
																}]
													})[0]

		print("A Training Set Object named " + str(params['training_set_name']) + " with reference: " + str(training_set_ref) + " was just made.")


	def checkUniqueColumn(self, uploaded_df, genome_label):

		if(uploaded_df[genome_label].is_unique):
			pass
		else:
			raise ValueError(str(genome_label) + "column is not unique")

	def findMissingGenomes(self, current_ws, uploaded_df):

		uploaded_df_columns = uploaded_df.columns
		all_genomes_workspace = self.ws_client.list_objects({'workspaces':[current_ws],'type':'KBaseGenomes.Genome'})

		#figure out if matching on Reference or Name and then find missing genomes
		if "Genome Reference" in uploaded_df_columns:
			genome_label = "Genome Reference"
			self.checkUniqueColumn(uploaded_df, genome_label)

			all_df_genome = uploaded_df[genome_label]
			all_refs2_workspace = [str(genome[0]) + "/" + str(genome[4]) for genome in all_genomes_workspace]
			all_refs3_workspace = [str(genome[0]) + "/" + str(genome[4]) + "/" + str(genome[6]) for genome in all_genomes_workspace]
			
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

		return (genome_label, all_df_genome, missing_genomes)

	def RASTAnnotateGenome(self, current_ws, input_genomes, output_genomes, output_genome_set_name):

		input_genomes_list = []
		for index, element in enumerate(output_genomes):
			input_genomes_list.append({"input_genome": element,
										"output_genome": element+".RAST"})

		params_RAST =	{
		"workspace": current_ws,
		"annotate_proteins_kmer_v2": 1,
		"annotate_proteins_similarity": 1,
		"call_features_CDS_glimmer3": 0,
		"call_features_CDS_prodigal": 0,
		"call_features_crispr": 0,
		"call_features_prophage_phispy": 0,
		"call_features_rRNA_SEED": 0,
		"call_features_repeat_region_SEED": 0,
		"call_features_strep_pneumo_repeat": 0,
		"call_features_strep_suis_repeat": 0,
		"call_features_tRNA_trnascan": 0,
		"call_pyrrolysoproteins": 0,
		"call_selenoproteins": 0,
		"genome_text": "",
		"input_genomes": input_genomes_list,
		"kmer_v1_parameters": 1,
		"output_genome": output_genome_set_name,
		"resolve_overlapping_features": 0,
		"retain_old_anno_for_hypotheticals": 0
		}
		
		#we don't do anything with the output but you can if you want to
		print(params_RAST)
		output = self.rast.annotate_genomes(params_RAST)

		if(output):
			pass
		else:
			print("output is: " + str(output))
			raise ValueError("for some reason unable to RAST Annotate Genome")


	def createListsForPredictionSet(self, current_ws, params, uploaded_df):
		(genome_label, all_df_genome, missing_genomes) = self.findMissingGenomes(current_ws, uploaded_df)
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

		if(params["annotate"]):
			
			#RAST Annotate the Genome
			output_genome_set_name = params['training_set_name'] + "_RAST_Genome_SET"
			self.RASTAnnotateGenome(current_ws, input_genome_references, input_genome_names, output_genome_set_name)

			#We know ahead of time that all names are just old names with .RAST appended to them
			RAST_genome_names = [genome_name + ".RAST" for genome_name in input_genome_names]
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
			<button class="tablinks" onclick="openTab(event, 'Main Report')" id="defaultOpen">Main Report</button>
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

		<div id="Main Report" class="tabcontent">
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

	def uploadHTMLContent(self, training_set_name, selected_file_name, missing_genomes, genome_label, phenotype, upload_table):
		
		file = open(os.path.join(self.scratch, 'forUpload', 'status.html'), "w")
		header = u"""
			<!DOCTYPE html>
			<html>
			<link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.21/css/jquery.dataTables.min.css">
			<body>

			<h2 style="text-align:center;"> Report: Upload Training Set Data </h2>
			<p>
			"""
		file.write(header)

		first_paragraph = u"""A Genome Classifier Training Set Object named """ + str(training_set_name) \
							+ """ was created and added to the workspace. Missing genomes (those that were \
							present in the selected file: """ + str(selected_file_name) + """, but not present in the staging area) \
							were the following: """ + str(missing_genomes)+ """ . """ + str(training_set_name) + """ was created \
							excluding the missing genomes.</p><p>"""
		file.write(first_paragraph)

		second_paragraph = u"""Below is a detailed table which shows """ + str(genome_label) + """ , whether it \
						was loaded into the workspace, its """ + str(phenotype)+ """, and if it is in training_set_name, its References, \
						its Evidence List. </p>"""
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

	def buildMainHTMLContent(self, main_report_df, genome_classifier_object_names, phenotype, best_classifier_type_nice):

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

			<h2 style="text-align:center;"> Report: Build Genome Classifier - Overview </h2>
			<p>
			"""
		file.write(header)

		first_paragraph = 	u"""The following Genome Categorizer Objects were created """ + str(genome_classifier_object_names) +\
							""" to classify genomes based on """ + str(phenotype) + """. Below we display a confusion matrix which \
							evaluates the performance of the selected classification algorithms. To do so, for each """ + str(phenotype) + """ \
							class we compute the percentage of genomes with a true label that get classified with a predicted label. Read more \
							about <a href="https://en.wikipedia.org/wiki/Confusion_matrix"> Confusion Matrices</a>. Given the magnitude of the \
							number of classes relative to the overall size of the training set, creating only one test set would lead to 
							inconclusive results. Instead, we use K-Fold (K=10) Cross Validation to ensure the quality of the model. Thus the below 
							confusion matrices represent the average percentages over all 10 folds.</p><p>
							"""
		file.write(first_paragraph)

		second_paragraph = 	u"""For each classification algorithm, we also provide the Precision, Recall, and F1-Score for each """ + str(phenotype) + """ \
							class. More information about these metrics can be found <a href="https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html#sklearn.metrics.precision_recall_fscore_support">
							here</a>.</p><p>
							"""
		file.write(second_paragraph)

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

			<h2 style="text-align:center;"> Report: Build Genome Classifier - Decision Tree Tuning </h2>
			<p>
			"""
		file.write(header)

		first_paragraph = 	u"""Since the feature space (functional role) is categorical, we further fine tune the Decision Tree \
							classification algorithm in particular to seek better metrics. We tune the Decision Tree based on two \
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

			<h2 style="text-align:center;"> Report: Predict Phenotype </h2>
			<p>
			"""
		file.write(header)

		first_paragraph = u"""The Genome Categorizer named """ + str(categorizer_name) \
							+ """ is being used to make predictions for  """ + str(phenotype)+ """ based on \
							""" + str(selection_attribute) + """. Missing genomes (those that were \
							present in the selected file: """ + str(selected_file_name) + """, but not present in the staging area) \
							were the following: """ + str(missing_genomes)+ """ ."""
		file.write(first_paragraph)

		second_paragraph = u"""Below is a detailed table which shows """ + str(genome_label) + """ , whether it \
						was loaded into the workspace, its """ + str(phenotype)+ """, and the probabiltiy of that \
						prediction.</p>"""
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



