import os
import uuid
import numpy as np
import pandas as pd


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
		# folder_name = "forUpload"
		# os.makedirs(os.path.join(self.scratch, folder_name))

		params["file_path"] = "/kb/module/data/RealData/GramDataEdit5.xlsx"
		uploaded_df = self.getUploadedFileAsDF(params["file_path"])
		(upload_table, classifier_training_set, missing_genomes, genome_label) = self.createAndUseListsForTrainingSet(current_ws, params, uploaded_df)

		self.uploadHTMLContent(params['training_set_name'], params["file_path"], missing_genomes, genome_label, params['phenotype'], upload_table)
		html_output_name = self.viewerHTMLContent("forUpload", status_view = True)
		
		return html_output_name, classifier_training_set

	#return html_output_name, classifier_info_list, weight_list
	def fullClassify(self, params, current_ws):
		pass

		"""
		should figure out ahead of time how many views to show
			ie. overview.html, dtt.html, ensemble
		"""

	#return html_output_name, predictions_mapping
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
																  'provenance': self.ctx.get('provenance')
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
		for index, element in enumerate(input_genomes):
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
		file = open(os.path.join(self.scratch, folder_name, 'viewer.html'), u"w")

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
		
		os.makedirs(os.path.join(self.scratch, 'forUpload'), exist_ok=True)
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
						prediction </p>"""
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

























