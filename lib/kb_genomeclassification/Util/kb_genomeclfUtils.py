import os
import uuid
import numpy as np
import pandas as pd

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
		os.makedirs(os.path.join(self.scratch, folder_name), exist_ok=True)

		params["file_path"] = "/kb/module/data/RealData/GramDataEdit5.xlsx"
		uploaded_df = self.getUploadedFileAsDF(params["file_path"])
		(upload_table, classifier_training_set, missing_genomes, genome_label) = self.createAndUseListsForTrainingSet(current_ws, params, uploaded_df)

		self.uploadHTMLContent(params['training_set_name'], params["file_path"], missing_genomes, genome_label, params['phenotype'], upload_table)
		html_output_name = self.viewerHTMLContent(folder_name, status_view = True)
		
		return html_output_name, classifier_training_set

	#return html_output_name, classifier_info_list, weight_list
	def fullClassify(self, params, current_ws):

		#double check that Ensemble is only called as specifid in the document
		if((params["ensemble_model"]!=None) and (classifier_to_run != "run_all")):
			raise ValueError("Ensemble Model will only be generated if Run All Classifiers is selected")


		#create folder for images and data
		folder_name = "forBuild"
		os.makedirs(os.path.join(self.scratch, folder_name), exist_ok=True)
		os.makedirs(os.path.join(self.scratch, folder_name, "images"), exist_ok=True)
		os.makedirs(os.path.join(self.scratch, folder_name, "data"), exist_ok=True)

		#unload the training_set_object
		#uploaded_df is four columns: Genome Name | Genome Reference | Phenotype | Phenotype Enumeration
		(phenotype, class_enumeration, uploaded_df, training_set_object_reference) = self.unloadTrainingSet(params['training_set_name'])

		#get functional_roles and make indicator matrix
		(indicator_matrix, master_role_list) = createIndicatorMatrix(uploaded_df, params["genome_attribute"])

		#split up training data
		splits = 2 #5
		whole_X = indicator_matrix[master_role_list].values
		whole_Y = uploaded_df["Phenotype Enumeration"].values
		(list_train_index, list_test_index) = getKSplits(splits, whole_X, whole_Y)

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
				'training_set_ref' : training_set_ref,
				'description' : params["description"]
				}

		dict_classification_report_dict = {}

		classifier_to_run = params["classifier_to_run"]
		if(classifier_to_run == "run_all"):

			list_classifier_types = ["k_nearest_neighbors", "gaussian_nb", "logistic_regression", "decision_tree_classifier", "support_vector_machine", "neural_network"]
			for classifier_type in list_classifier_types:
				current_classifier_object = {	"classifier_to_execute": getCurrentClassifierObject(classifier_type),
												"classifier_type": classifier_type,
												"classifier_name": params["classifier_object_name"] + "_" + classifier_type
											}

				#this is a dictionary containing 'class 0': {'precision': 0.5, 'recall': 1.0, 'f1-score': 0.2}, 'accuracy'
				classification_report_dict = self.executeClassifier(current_ws, common_classifier_information, current_classifier_object, folder_name)
				dict_classification_report_dict[classifier_type] = classification_report_dict


				#handle Decision Tree Case
				(ddt_dict_classification_report_dict, top_20) = self.tuneDecisionTree(current_ws, common_classifier_information, params["classifier_object_name"], folder_name)
				dict_classification_report_dict["decision_tree_classifier_gini"] = ddt_dict_classification_report_dict["decision_tree_classifier_gini"]
				dict_classification_report_dict["decision_tree_classifier_entropy"] = ddt_dict_classification_report_dict["decision_tree_classifier_entropy"]
			
			#handle case for ensemble
		else:
			current_classifier_object = {	"classifier_to_execute": getCurrentClassifierObject(classifier_to_run),
											"classifier_type": classifier_to_run,
											"classifier_name": params["classifier_object_name"] + "_" + classifier_to_run
										}

			classification_report_dict = self.executeClassifier(current_ws, common_classifier_information, current_classifier_object)
			dict_classification_report_dict[classifier_to_run] = classification_report_dict

			if(classifier_to_run == "decision_tree_classifier"):
				(ddt_dict_classification_report_dict, top_20) = self.tuneDecisionTree(current_ws, common_classifier_information, params["classifier_object_name"], folder_name)
				dict_classification_report_dict["decision_tree_classifier_gini"] = ddt_dict_classification_report_dict["decision_tree_classifier_gini"]
				dict_classification_report_dict["decision_tree_classifier_entropy"] = ddt_dict_classification_report_dict["decision_tree_classifier_entropy"]

			else:
				#create folder only for Main


		#make the storage folders
		#overview.html, dtt.html, ensemble.html

		#generate table from dict_classification_report_dict
		#write self.buildMainHTMLContent
		#write self.buildDTTHTMLContent

		#return html_output_name, predictions_mapping

		else:
			ensemble_params["flatten_transform"] = self.str_to_bool(ensemble_params["flatten_transform"])

		return ensemble_params

	def executeClassifier(self, current_ws, common_classifier_information, current_classifier_object, folder_name):
		
		matrix_size = len(common_classifier_information["class_list_mapping"])
		cnf_matrix_proportion = np.zeros(shape=(matrix_size, matrix_size))
		
		for c in range(common_classifier_information["splits"]):
			X_train = common_classifier_information["whole_X"][common_classifier_information["list_train_index"][c]]
			y_train = common_classifier_information["whole_Y"][common_classifier_information["list_train_index"][c]]
			X_test = common_classifier_information["whole_X"][common_classifier_information["list_test_index"][c]]
			y_test = common_classifier_information["whole_Y"][common_classifier_information["list_test_index"][c]]

			classifier.fit(X_train, y_train)
			y_pred = classifier.predict(X_test)

			cnf = confusion_matrix(y_test, y_pred, lables=list(common_classifier_information["class_enumeration"].values()))
			cnf_f = cnf.astype('float') / cnf.sum(axis=1)[:, np.newaxis]
			for i in range(len(cnf)):
				for j in range(len(cnf)):
					cnf_matrix_proportion[i][j] += cnf_f[i][j]

		#get statistics for the last case made
		#diagonal entries of cm are the accuracies of each class
		target_names = list(common_classifier_information["class_enumeration"].keys())
		classification_report_dict = classification_report(y_test, y_pred, target_names=target_names, output_dict = True)

		#save down classifier object in pickle format
		pickle_out = open(os.path.join(self.scratch, folder_name, "data", current_classifier_object["classifier_name"] + ".pickle"), "w")
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
     
		cm = np.round(cnf_matrix_proportion/splits*100.0,1)
		title = "CM: " + current_classifier_object["classifier_type"]
		#classes = list(common_classifier_information["class_enumeration"].keys())
     	self.plot_confusion_matrix(cm, title, current_classifier_object["classifier_name"], list(common_classifier_information["class_list_mapping"].keys()), folder_name)

     	return classification_report_dict


	def tuneDecisionTree(self, current_ws, common_classifier_information, classifier_object_name, folder_name):

		iterations = 13
		ddt_dict_classification_report_dict = {}

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
			validate_std.append(np.std(validate_score))

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
										"classifier_name": classifier_object_name + "_" + classifier_type + "_gini" 
									}

		#this is a dictionary containing 'class 0': {'precision': 0.5, 'recall': 1.0, 'f1-score': 0.2}, 'accuracy'
		classification_report_dict = self.executeClassifier(current_ws, common_classifier_information, current_classifier_object, folder_name)
		ddt_dict_classification_report_dict["decision_tree_classifier_gini"] = classification_report_dict


		
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
			validate_std.append(np.std(validate_score))

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
										"classifier_name": classifier_object_name + "_" + classifier_type + "_entropy" 
									}

		#this is a dictionary containing 'class 0': {'precision': 0.5, 'recall': 1.0, 'f1-score': 0.2}, 'accuracy'
		classification_report_dict = self.executeClassifier(current_ws, common_classifier_information, current_classifier_object, folder_name)
		ddt_dict_classification_report_dict["decision_tree_classifier_entropy"] = classification_report_dict

		if best_gini_accuracy_score > best_entropy_accuracy_score:
			top_20 = self.tree_code(DecisionTreeClassifier(random_state=0, max_depth=best_gini_depth, criterion=u'gini'), common_classifier_information)
		else:
			top_20 = self.tree_code(DecisionTreeClassifier(random_state=0, max_depth=best_entropy_depth, criterion=u'entropy'), common_classifier_information)

		return (ddt_dict_classification_report_dict, top_20)

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
		for i in range(len(class_list)):
			color_set.append('%.4f'%np.random.random() + " " + '%.4f'%np.random.random()+ " " + '0.900')

		for current_class, current_color in zip(class_list, color_set):
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

		training_set_object = self.ws_client.get_objects2({'objects' : [{'workspace':current_ws, 'name': training_set_name}]})
		phenotype = training_set_object["classification_type"]
		classes_sorted = training_set_object["classes"]

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

		uploaded_data = {	"Genome Name": _names,
						"Geomne Reference": _references,
						"Phenotype": _phenotypes,
						"Phenotype Enumeration": _enumeration}

		uploaded_df = pd.DataFrame(data=upload_data)

		return(phenotype, class_enumeration, uploaded_df, training_set_object_reference)

	def createIndicatorMatrix(self, uploaded_df, genome_attribute):
		genome_references = uploaded_df["Genome Reference"].to_list()

		if "functional_roles" == genome_attribute:
			
			ref_to_role = {}
			master_role_set = {}

			for genome_ref in genome_references:
				genome_object_data = self.ws_client.get_objects2({'objects':[{'ref': 'genome_ref'}]})['data'][0]['data']

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
						list_functional_roles.append(functional_role[function_str])
					else:
						#apparently some function list just don't have functions...
						pass

				#create a mapping from genome_ref to all of its functional roles
				ref_to_role[genome_ref] = list_functional_roles

				#keep updateing a set of all functional roles seen so far
				master_role_set.union(set(list_functional_roles))

			#we are done looping over all genomes
			master_role_list = sorted(list(master_role_set))
			master_role_enumeration = enumerate(master_role_set)
			ref_to_indication = {}

			#make indicator rows for each 
			for genome_ref in genome_references:
				set_functional_roles = set(ref_to_role[genome_ref])
				matching_index = [i for i, role in master_role_enumeration if role in set_functional_roles] 

				indicators = np.zeros(len(master_role_list))
				indicators[np.array(matching_index)] = 1

				ref_to_indication[genome_ref] = indicators.astype(int)


			indicator_matrix = pd.DataFrame(data = ref_to_indication, orient='index', columns = master_role_list).reset_index().rename(columns={"index":"Genome Reference"})
		
			return (indicator_matrix, master_role_list)
		else:
			raise ValueError("Only classifiers based on functional roles have been impliemented please check back later")



	def getKSplits(self, splits, whole_X, whole_Y):

		#This cross-validation object is a variation of KFold that returns stratified folds. The folds are made by preserving the percentage of samples for each class.
		skf = StratifiedKFold(n_splits=splits, random_state=0, shuffle=True)
		for train_idx, test_idx in skf.split(whole_X, whole_Y):
			list_train_index.append(train_idx)
			list_test_index.append(test_idx)

		return (list_train_index, list_test_index)

	def fullPredict(self, params, current_ws):

		# #Load Information from Categorizer 
		# categorizer_object = ws_client.get_objects2({'objects' : [{'workspace':current_ws, 'name':params['categorizer_name']}]})

		# categorizer_handle_ref = categorizer_object[0]['data']['classifier_handle_ref']
		# categorizer_file_path = self._download_shock(categorizer_handle_ref)

		# master_feature_list = categorizer_object[0]['data']['attribute_data']
		# class_to_index_mapping = categorizer_object[0]['data']['class_list_mapping']

		# current_categorizer = pickle.load(open(categorizer_file_path, "rb"))


		# #Load Information from UploadedFile
		# uploaded_df = getUploadedFileAsDF(params["file_path"])
		# (missing_genomes, genome_label, _genome_df, _in_workspace, _list_genome_name, _list_genome_ref) = createListsForPredictionSet(current_ws, params, uploaded_df)
		# feature_matrix = self.getFeatureMatrix(stuff goes here)


		# #Make Predictions on uploaded file
		# predictions_numerical = current_categorizer.predict(feature_matrix)
		# predictions_phenotype = #map numerical to phenotype
		# prediction_probabilities = current_categorizer.predict_proba(feature_matrix)


		# #Lists to use for report
		# _prediction_phenotype = []
		# _prediction_probabilities = []
		# index = 0

		# #all lists for callback structure and training set object
		# _list_prediction_phenotype = []
		# _list_prediction_probabilities = []

		# for genome in _genome_df:
		# 	if(genome not in missing_genomes):
		# 		_prediction_phenotype.append(predictions_phenotype[index])
		# 		_prediction_probabilities.append(prediction_probabilities[index])
		# 		index +=1

		# 		_list_prediction_phenotype.append(predictions_phenotype[index])
		# 		_list_prediction_probabilities.append(prediction_probabilities[index])

		# 	else:
		# 		_prediction_phenotype.append("N/A")
		# 		_prediction_probabilities.append("N/A")


		# #construct classifier_training_set mapping
		# prediction_set = {}
		
		# for index, curr_genome_ref in enumerate(_list_genome_ref):
		# 	prediction_set[curr_genome_ref] = { 'genome_name': _list_genome_name[index],
		# 										'genome_ref': curr_genome_ref,
		# 										'phenotype': _list_prediction_phenotype[index],
		# 										'prediction_probabilities': _list_prediction_probabilities[index]
		# 										}

		# report_table = pd.DataFrame.from_dict({	genome_label: _genome_df,
		# 										"In Workspace": _in_workspace,
		# 										"Phenotype": _prediction_phenotype,
		# 										"Probability": _prediction_probabilities
		# 									 	})



		# self.html_report_3(missingGenomes, params['phenotypeclass'])
		# html_output_name = self.html_nodual("forSecHTML")
		#handle making a report in html

		return html_output_name, predictions_mapping


	def generateHTMLReport(self, current_ws, folder_name, single_html_name, description, for_build_classifier = False):

		#folder_name = forUpload forBuild forPredict

		report_shock_id = self.dfu.file_to_shock({'file_path': os.path.join(self.scratch, folder_name),'pack': 'zip'})['shock_id']

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

	def getUploadedFileAsDF(self, file_path):

		if file_path.endswith('.xlsx'):
			uploaded_df = pd.read_excel(os.path.join(os.path.sep,"staging",file_path), dtype=str)
		elif file_path.endswith('.csv'):
			uploaded_df = pd.read_csv(os.path.join(os.path.sep,"staging",file_path), header=0, dtype=str)
		elif file_path.endswith('.tsv'):
			uploaded_df = pd.read_csv(os.path.join(os.path.sep,"staging",file_path), sep='\t', header=0, dtype=str)
		else:
			raise ValueError('The following file type is not accepted, must be .xlsx, .csv, .tsv')

		self.checkValidFile(uploaded_df)
		return uploaded_df

	def checkValidFile(self, uploaded_df):
		
		uploaded_df_columns = uploaded_df.columns
		if (("Genome Name" in uploaded_df_columns) or ("Genome Reference" in uploaded_df_columns)) and ("Phenotype" in uploaded_df_columns):
			pass
		else:
			raise ValueError('File must include Genome Name/Genome Reference and Phenotype as columns')

	def createAndUseListsForTrainingSet(self, current_ws, params, uploaded_df):
		
		(genome_label, all_df_genome, missing_genomes) = self.findMissingGenomes( current_ws, uploaded_df)

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
		(genome_label, all_df_genome, missing_genomes) = findMissingGenomes(current_ws, uploaded_df)

		# all lists needed for report
		_genome_df = []
		_in_workspace = []

		#all lists for callback structure and prediction object
		_list_genome_name = []
		_list_genome_ref = []

		for genome in all_df_genome:	
			#genome is present in workspace
			if(genome not in missing_genomes):

				#figure out genome_name and genome_ref
				genome_name = genome if genome_label == "Genome Name" else str(self.ws_client.get_objects2({'objects' : [{'ref':genome}]})['data'][0]['info'][1])	
				if(genome_label == "Genome Reference"):
					genome_ref = genome
				else:
					meta_data = self.ws_client.get_objects2({'objects' : [{'workspace':current_ws, 'name': genome}]})['data'][0]['info']
					genome_ref = str(meta_data[6]) + "/" + str(meta_data[0]) + "/" + str(meta_data[4])


				#indicates that users wants us to RAST annotate the Genomes
				if(params["annotate"]):
					rast_genome_name = genome_name + ".RAST"
					self.RASTAnnotateGenome(current_ws, genome_name, rast_genome_name)
					_genome_df.append(rast_genome_name)

					RAST_meta_data = self.ws_client.get_objects2({'objects' : [{'workspace':current_ws, 'name': rast_genome_name}]})['data'][0]['info']
					rast_genome_ref = str(RAST_meta_data[6]) + "/" + str(RAST_meta_data[0]) + "/" + str(RAST_meta_data[4])
					
					#in the training set, set the genome_name and genome_ref with rast
					_list_genome_name.append(rast_genome_name)
					_list_genome_ref.append(rast_genome_ref)

				else:
					_genome_df.append(genome)
					
					#in the training set, set the genome_name and genome_ref exactly what the user passed in
					_list_genome_name.append(genome_name)
					_list_genome_ref.append(genome_ref)

				#lists for report
				_in_workspace.append("True")

			else:
				_genome_df.append(genome)
				_in_workspace.append("False")

		return (missing_genomes, genome_label, _genome_df, _in_workspace, _list_genome_name, _list_genome_ref)


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
			str_button = u"""
			<button class="tablinks" onclick="openTab(event, 'Decision Tree Tuning')">Decision Tree Tuning</button>
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
		  <iframe src="dtt.html" style="height:100vh; width:100%; border: hidden;" ></iframe>
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



	def predictHTMLContent(self, categorizer_name, phenotype, selection_attribute, selected_file_name, missing_genomes, genome_label, predict_table):
		
		# file = open(os.path.join(self.scratch, 'forPredict', 'status.html'), u"w")
		# header = u"""
		# 	<!DOCTYPE html>
		# 	<html>
		# 	<link rel="stylesheet" type="text/css" href="https://cdn.datatables.net/1.10.21/css/jquery.dataTables.min.css">
		# 	<body>

		# 	<h2 style="text-align:center;"> Report: Predict Phenotype </h2>
		# 	<p>
		# 	"""
		# file.write(header)

		# first_paragraph = u"""The Genome Categorizer named """ + str(categorizer_name) \
		# 					+ """ is being used to make predictions for  """ + str(phenotype)+ """ based on \
		# 					""" + str(selection_attribute) + """. Missing genomes (those that were \
		# 					present in the selected file: """ + str(selected_file_name) + """, but not present in the staging area) \
		# 					were the following: """ + str(missing_genomes)+ """ ."""
		# file.write(first_paragraph)

		# second_paragraph = u"""Below is a detailed table which shows """ + str(genome_label) + """ , whether it \
		# 				was loaded into the workspace, its """ + str(phenotype)+ """, and the probabiltiy of that \
		# 				prediction </p>"""
		# file.write(second_paragraph)

		# predict_table_html = predict_table.to_html(index=False, table_id="predict_table", justify='center')
		# file.write(predict_table_html)

		# scripts = u"""</body>

		# 	<script type="text/javascript" src="https://code.jquery.com/jquery-3.5.1.js"></script>
		# 	<script type="text/javascript" src="https://cdn.datatables.net/1.10.21/js/jquery.dataTables.min.js"></script>
		# 	<script type="text/javascript">
		# 	$(document).ready(function() {
		# 		$('#predict_table').DataTable( {
		# 			"scrollY":        "500px",
		# 			"scrollCollapse": true,
		# 			"paging":         false
		# 		} );
		# 	} );
		# 	</script>
		# 	</html>"""
		# file.write(scripts)
		# file.close()

























