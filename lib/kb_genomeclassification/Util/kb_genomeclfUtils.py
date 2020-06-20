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

	#return htmloutput_name, classifier_training_set_mapping
	def fullUpload(self, params, current_ws):
	
		uploaded_df = getUploadedFileAsDF(params["file_path"])
		(report_table, classifier_training_set) = createAndUseListsForTrainingSet(current_ws, params, uploaded_df)

		#self.html_report_0(missingGenomes, params['phenotypeclass'])
		#htmloutput_name = self.html_nodual("forZeroHTML")
		#handle making report in html

		return htmloutput_name, classifier_training_set

	#return htmloutput_name, classifier_info_list, weight_list
	def fullClassify(self, params, current_ws):
		pass

	#return htmloutput_name, predictions_mapping
	def fullPredict(self, params, current_ws):

		#Load Information from Categorizer 
		categorizer_object = ws_client.get_objects2({'objects' : [{'workspace':current_ws, 'name':params['categorizer_name']}]})

		categorizer_handle_ref = categorizer_object[0]['data']['classifier_handle_ref']
		categorizer_file_path = self._download_shock(categorizer_handle_ref)

		master_feature_list = categorizer_object[0]['data']['attribute_data']
		class_to_index_mapping = categorizer_object[0]['data']['class_list_mapping']

		current_categorizer = pickle.load(open(categorizer_file_path, "rb"))


		#Load Information from UploadedFile
		uploaded_df = getUploadedFileAsDF(params["file_path"])
		(missing_genomes, genome_label, _genome_df, _in_workspace, _list_genome_name, _list_genome_ref) = createListsForPredictionSet(current_ws, params, uploaded_df)
		feature_matrix = self.getFeatureMatrix(stuff goes here)


		#Make Predictions on uploaded file
		predictions_numerical = current_categorizer.predict(feature_matrix)
		predictions_phenotype = #map numerical to phenotype
		prediction_probabilities = current_categorizer.predict_proba(feature_matrix)


		#Lists to use for report
		_prediction_phenotype = []
		_prediction_probabilities = []
		index = 0

		#all lists for callback structure and training set object
		_list_prediction_phenotype = []
		_list_prediction_probabilities = []

		for genome in _genome_df:
			if(genome not in missing_genomes):
				_prediction_phenotype.append(predictions_phenotype[index])
				_prediction_probabilities.append(prediction_probabilities[index])
				index +=1

				_list_prediction_phenotype.append(predictions_phenotype[index])
				_list_prediction_probabilities.append(prediction_probabilities[index])

			else:
				_prediction_phenotype.append("N/A")
				_prediction_probabilities.append("N/A")


		#construct classifier_training_set mapping
		prediction_set = {}
		
		for index, curr_genome_ref in enumerate(_list_genome_ref):
			prediction_set[curr_genome_ref] = { 'genome_name': _list_genome_name[index],
												'genome_ref': curr_genome_ref,
												'phenotype': _list_prediction_phenotype[index],
												'prediction_probabilities': _list_prediction_probabilities[index]
												}

		report_table = pd.DataFrame.from_dict({	genome_label: _genome_df,
												"In Workspace": _in_workspace,
												"Phenotype": _prediction_phenotype,
												"Probability": _prediction_probabilities
											 	})



		# self.html_report_3(missingGenomes, params['phenotypeclass'])
		# htmloutput_name = self.html_nodual("forSecHTML")
		#handle making a report in html

		return htmloutput_name, predictions_mapping

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
			uploaded_df = pd.read_excel(os.path.join(os.path.sep,"staging",file_path))
		elif file_path.endswith('.csv'):
			uploaded_df = pd.read_csv(os.path.join(os.path.sep,"staging",file_path), header=0)
		elif file_path.endswith('.tsv'):
			uploaded_df = pd.read_csv(os.path.join(os.path.sep,"staging",file_path), sep='\t', header=0)
		else:
			raise ValueError('The following file type is not accepted, must be .xlsx, .csv, .tsv')

		checkValidFile(uploaded_df)
		return uploaded_df

	def checkValidFile(self, uploaded_df):
		
		uploaded_df_columns = uploaded_df.columns
		if (("Genome Name" in uploaded_df_columns) or ("Genome Reference" in uploaded_df_columns)) and ("Phenotype" in uploaded_df_columns):
			pass
		else:
			raise ValueError('File must include Genome Name/Genome Reference and Phenotype as columns')

	def createAndUseListsForTrainingSet(self, current_ws, params, uploaded_df):
		
		(genome_label, all_df_genome, missing_genomes) = findMissingGenomes( current_ws, uploaded_df)

		# all lists needed for report
		_genome_df = []
		_in_workspace = []
		_phenotype = []
		_in_training_set = []

		_references = []
		has_references = True if "References" in uploaded_df_columns else False
		_evidence_types = []
		has_evidence_types = True if "Evidence Types" in uploaded_df_columns else False


		#all lists for callback structure and training set object
		_list_genome_name = []
		_list_genome_ref = []
		_list_phenotype = []
		_list_genome_id = [] #for now always ""
		_list_references = []
		_list_evidence_types = []

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
					RASTAnnotateGenome(self, current_ws, genome_name, rast_genome_name)
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
				_phenotype.append(uploaded_df.loc[uploaded_df[genome_label] == genome]["Phenotype"][0])
				_in_training_set.append("True")

				#lists for callback structure and training set object
				_list_phenotype.append(uploaded_df.loc[uploaded_df[genome_label] == genome]["Phenotype"][0])
				_list_genome_id.append("") #setting this to "" just for now

				#non-missing genomes add references
				if(has_references):
					_list_references.append(uploaded_df.loc[uploaded_df[genome_label] == genome]["References"][0].split(";"))
				else:
					_list_references.append([]) #add empty list

				#non-missing genomes add evidence list
				if(has_evidence_types):
					_list_evidence_types.append(uploaded_df.loc[uploaded_df[genome_label] == genome]["Evidence Types"][0].split(";"))
				else:
					_list_evidence_types.append([]) #add empty list

			else:
				_genome_df.append(genome)
				_in_workspace.append("False")
				_phenotype.append(uploaded_df.loc[uploaded_df[genome_label] == genome]["Phenotype"][0])
				_in_training_set.append("False")

			#add references
			if(has_references):
				_references.append(uploaded_df.loc[uploaded_df[genome_label] == genome]["References"][0].split(";"))
			else:
				_references.append("")

			#add evidence list
			if(has_evidence_types):
				_evidence_types.append(uploaded_df.loc[uploaded_df[genome_label] == genome]["Evidence Types"][0].split(";"))
			else:
				_evidence_types.append("")


		#construct classifier_training_set mapping
		classifier_training_set = {}
		
		for index, curr_genome_ref in enumerate(_list_genome_ref):
			classifier_training_set[curr_genome_ref] = { 	'genome_name': _list_genome_name[index],
															'genome_ref': curr_genome_ref,
															'phenotype': _list_phenotype[index],
															'references': _list_references[index],
															'evidence_types': _list_evidence_types[index]
														}
		
		report_table = pd.DataFrame.from_dict({	genome_label: _genome_df,
												"In Workspace": _in_workspace,
												"Phenotype": _phenotype,
												"In Training Set": _in_training_set
											 	})

		createTrainingSetObject(current_ws, params, _list_genome_name, _list_genome_ref, _list_phenotype, _list_genome_id, _list_references, _list_evidence_types)
		return (report_table, classifier_training_set)

	def createTrainingSetObject(self, current_ws, params, _list_genome_name, _list_genome_ref, _list_phenotype, _list_genome_id, _list_references, _list_evidence_types):

		classification_data = []

		for index, curr_genome_ref in enumerate(_list_genome_ref):
			classification_data.append({ 	'genome_name': _list_genome_name[index],
											'genome_ref': curr_genome_ref,
											'genome_classification': _list_phenotype[index],
											'genome_id': _list_genome_id[index],
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

		print("A Training Set Object named " + str(params['training_set_name']) + " with reference: " + str(training_set_ref) " was just made.")


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
			checkUniqueColumn(uploaded_df, genome_label)

			all_df_genome = uploaded_df[genome_label]
			all_refs2_workspace = [str(genome[0]) + "/" + str(genome[4]) for genome in all_genomes_workspace]
			all_refs3_workspace = [str(genome[0]) + "/" + str(genome[4]) + "/" + str(genome[6]) for genome in all_genomes_workspace]
			
			missing_genomes = []
			for ref in all_df_genome:
				if((ref not in all_refs2_workspace) and (ref not in all_refs3_workspace)):
					missing_genomes.append(ref)

		else:
			genome_label = "Genome Name"
			checkUniqueColumn(uploaded_df, genome_label)

			all_df_genome = uploaded_df[genome_label]
			all_names_workspace = [str(genome[1]) for genome in all_genomes_workspace]
			missing_genomes = list(set(all_df_genome).difference(set(all_names_workspace)))

		return (genome_label, all_df_genome, missing_genomes)

	def RASTAnnotateGenome(self, current_ws, genome_name, rast_genome_name):

		params_RAST =	{
		"workspace": current_ws,
		"input_genome": genome_name,
		"output_genome": rast_genome_name,
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

		#we don't do anything with the output but you can if you want to
		output = self.rast.annotate_genome(params_RAST)

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
					RASTAnnotateGenome(self, current_ws, genome_name, rast_genome_name)
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

