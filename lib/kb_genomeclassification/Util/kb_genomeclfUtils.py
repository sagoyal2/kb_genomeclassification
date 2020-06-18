import os
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


	#return htmloutput_name, classifier_info_list, weight_list
	def fullClassify(self, params, current_ws):
		pass

	#return htmloutput_name, predictions_mapping
	def fullPredict(self, params, current_ws):
		pass



	###

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

	def createGenomeClassifierTrainingSet(self, current_ws, uploaded_df):

		"""
		first decide to get references or names
		for references: self.ws_client.list_objects({'workspaces':[current_ws],'type':'KBaseGenomes.Genome'}) [6][4][0]
		for names: [self.ws_client.list_objects({'workspaces':[current_ws],'type':'KBaseGenomes.Genome'}) [1]


		missing = differences user - all in workspace

		there is also classifier_training_set_mapping
		split references, and evidence on ;		mystr.split(";")



		"""
		
		uploaded_df_columns = uploaded_df.columns
		all_genomes_workspace = self.ws_client.list_objects({'workspaces':[current_ws],'type':'KBaseGenomes.Genome'})

		if "Genome Reference" in uploaded_df_columns:
			genomeLabel = "Genome Reference"
			all_df_genome = uploaded_df[genomeLabel]
			all_refs2_workspace = [str(genome[0]) + "/" + str(genome[4]) for genome in all_genomes_workspace]
			all_refs3_workspace = [str(genome[0]) + "/" + str(genome[4]) + "/" + str(genome[6]) for genome in all_genomes_workspace]
			
			missing_genomes = []
			for ref in all_df_genome:
				if((ref not in all_refs2_workspace) and (ref not in all_refs3_workspace)):
					missing_genomes.append(ref)

		else:
			genomeLabel = "Genome Name"
			all_df_genome = uploaded_df[genomeLabel]
			all_names_workspace = [str(genome[1]) for genome in all_genomes_workspace]
			missing_genomes = list(set(all_df_genome).difference(set(all_names_workspace)))


		#indicates that uses wants us to RAST annotate the Genomes
		if(params["annotate"]):

		phenotype = uploaded_df['Phenotype']
		classifier_training_set_mapping = {}

		for genome in all_df_genome:

			#genome is present in workspace
			if(genome not in missing_genomes):



			else:



		inKBASE.append(listGNames[index])
		inKBASE_Classification.append(listClassification[index])

		all_genome_ID.append(listGNames[index])
		loaded_Narrative.append(["Yes"])
		all_Genome_Classification.append(listClassification[index])
		add_trainingSet.append(["Yes"])

		eachGenomeDict['genome_name'] = listGNames[index]
		eachGenomeDict['genome_ref'] = str(self.ws_client.get_objects([{'workspace':current_ws, 'name': eachGenomeDict['genome_name'] }])[0]['refs'][0])
		eachGenomeDict['phenotype'] = listClassification[index]
		eachGenomeDict['load_status'] = 1
		eachGenomeDict['RAST_annotation_status'] = 1
		
		classifier_training_set_mapping[eachGenomeDict['genome_ref']] = eachGenomeDict



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
		for string, index in zip(listGNames, range(len(listGNames))):
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
			ref_names = []
			for index in range(len(listGNames)):
				try:
					if(Annotated==1):
						position = list_allGenomesinWS.index(listGNames[index])
						ref_names.append(str(self.ws_client.get_objects([{'workspace':current_ws, 'name': listGNames[index] }])[0]['refs'][0]))
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
								ref_names.append(str(self.ws_client.get_objects([{'workspace':current_ws, 'name': listGNames[index]+".RAST" }])[0]['refs'][0]))

							except:
								print (listGNames[index])
								print ('The above Genome does not exist in workspace')
								missingGenomes.append(listGNames[index])
								ref_names.append(str(self.ws_client.get_objects([{'workspace':current_ws, 'name': listGNames[index] }])[0]['refs'][0]))

				except:
					print (listGNames[index])
					print ('The above Genome does not exist in workspace')
					missingGenomes.append(listGNames[index])
					ref_names.append(str(self.ws_client.get_objects([{'workspace':current_ws, 'name': listGNames[index] }])[0]['refs'][0]))

			print(inKBASE)
			print(missingGenomes)

			return (missingGenomes, inKBASE, ref_names)

		listClassification = just_DF['Classification']

		classifier_training_set_mapping = {}

		#self.ws_client.get_objects([{'workspace':current_ws, 'name':'357804.5'}])[0]['refs'][0]

		for index in range(len(listGNames)):

			eachGenomeDict = {}

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

					eachGenomeDict['genome_name'] = listGNames[index]+".RAST"
					eachGenomeDict['genome_ref'] = str(self.ws_client.get_objects([{'workspace':current_ws, 'name': eachGenomeDict['genome_name'] }])[0]['refs'][0])
					eachGenomeDict['phenotype'] = listClassification[index]
					eachGenomeDict['load_status'] = 1
					eachGenomeDict['RAST_annotation_status'] = 1

				else:
					# you will end up with case where the genomes will be RAST annotated but not have .RAST attached to it

					inKBASE.append(listGNames[index])
					inKBASE_Classification.append(listClassification[index])

					all_genome_ID.append(listGNames[index])
					loaded_Narrative.append(["Yes"])
					all_Genome_Classification.append(listClassification[index])
					add_trainingSet.append(["Yes"])

					eachGenomeDict['genome_name'] = listGNames[index]
					eachGenomeDict['genome_ref'] = str(self.ws_client.get_objects([{'workspace':current_ws, 'name': eachGenomeDict['genome_name'] }])[0]['refs'][0])
					eachGenomeDict['phenotype'] = listClassification[index]
					eachGenomeDict['load_status'] = 1
					eachGenomeDict['RAST_annotation_status'] = 1

					classifier_training_set_mapping[eachGenomeDict['genome_ref']] = eachGenomeDict


			except:
				print (listGNames[index])
				print ('The above Genome does not exist in workspace')
				missingGenomes.append(listGNames[index])

				all_genome_ID.append(listGNames[index])
				loaded_Narrative.append(["No"])
				all_Genome_Classification.append(listClassification[index])
				add_trainingSet.append(["No"])

				eachGenomeDict['genome_name'] = listGNames[index]
				eachGenomeDict['genome_ref'] = "None"
				eachGenomeDict['phenotype'] = listClassification[index]
				eachGenomeDict['load_status'] = 0
				eachGenomeDict['RAST_annotation_status'] = 0

			classifier_training_set_mapping[eachGenomeDict['genome_ref']] = eachGenomeDict
			del eachGenomeDict

		four_columns = pd.DataFrame.from_dict({'Genome Id': all_genome_ID, 'Loaded in the Narrative': loaded_Narrative, 'Classification' : all_Genome_Classification, 'Added to Training Set' : add_trainingSet})
		four_columns = four_columns[['Genome Id', 'Loaded in the Narrative', 'Classification', 'Added to Training Set']]

		old_width = pd.get_option('display.max_colwidth')
		pd.set_option('display.max_colwidth', -1)
		four_columns.to_html(os.path.join(self.scratch, 'forZeroHTML', 'four_columns.html'), index=False, justify='center', table_id = "four_columns", classes =["table", "table-striped", "table-bordered"])
		pd.set_option('display.max_colwidth', old_width)


		print("done")

		# typedef structure {
        # string phenotype;
        # string genome_name;
        # string genome_ref;
        # int load_status;
        # int RAST_annotation_status;
    	# 	}	 ClassifierTrainingSetOut;
		# mapping <string genome_id,ClassifierTrainingSetOut> classifier_training_set;

		return (missingGenomes, inKBASE, inKBASE_Classification, classifier_training_set_mapping)



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












