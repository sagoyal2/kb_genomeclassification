/*
A KBase module: kb_genomeclassification
This module build a classifier and predict phenotypes based on the classifier Another line
*/

module kb_genomeclassification {

    /* typedef string genome_id;
       typedef string phenotype; */

    /* "True" or "False" */
    typedef string boolean;

    typedef structure {
        string phenotype;
        string genome_name;
    } ClassifierTrainingSet;

    typedef structure {
        string penalty;
        boolean dual;
        float lr_tolerance;
        float lr_C;
        boolean fit_intercept;
        float intercept_scaling;
        string lr_class_weight;
        int lr_random_state;
        string lr_solver;
        int lr_max_iter;
        string multi_class;
        boolean lr_verbose;
        int lr_warm_start;
        int lr_n_jobs;
    } LogisticRegressionOptions;

    typedef structure {
        string criterion;
        string splitter;
        int max_depth;
        int min_samples_split;
        int min_samples_leaf;
        float min_weight_fraction_leaf;
        string max_features;
        int dt_random_state;
        int max_leaf_nodes;
        float min_impurity_decrease;
        string dt_class_weight;
        string presort;
    } DecisionTreeClassifierOptions;

    typedef structure {
       string priors;
    } GaussianNBOptions;

    typedef structure {
        int n_neighbors;
        string weights;
        string algorithm;
        int leaf_size;
        int p;
        string metric;
        string metric_params;
        int knn_n_jobs;
    } KNearestNeighborsOptions;

    typedef structure {
        float svm_C;
        string kernel;
        int degree;
        string gamma;
        float coef0;
        boolean probability;
        boolean shrinking;
        float svm_tolerance;
        float cache_size;
        string svm_class_weight;
        boolean svm_verbose;
        int svm_max_iter;
        string decision_function_shape;
        int svm_random_state;
    } SupportVectorMachineOptions;

    typedef structure {
        string hidden_layer_sizes;
        string activation;
        string mlp_solver;
        float alpha;
        string batch_size;
        string learning_rate;
        float learning_rate_init;
        float power_t;
        int mlp_max_iter;
        boolean shuffle;
        int mlp_random_state;
        float mlp_tolerance;
        boolean mlp_verbose;
        boolean mlp_warm_start;
        float momentum;
        boolean nesterovs_momentum;
        boolean early_stopping;
        float validation_fraction;
        float beta_1;
        float beta_2;
        float epsilon;
    } NeuralNetworkOptions;

    typedef structure {
        int k_nearest_neighbors_box;
        int gaussian_nb_box;
        int logistic_regression_box;
        int decision_tree_classifier_box;
        int support_vector_machine_box;
        int neural_network_box;
        string voting;
        string en_weights;
        int en_n_jobs;
        boolean flatten_transform;
    } EnsembleModelOptions;

    typedef structure {
        string phenotypeclass;
        string attribute;
        string workspace;
        string trainingset_name;
        mapping <string genome_id,ClassifierTrainingSet> classifier_training_set;
        string classifier_out;
        string target;
        string description;
        string classifier;
        string shock_id;
        string list_name;
        int save_ts;
        LogisticRegressionOptions logistic_regression;
        DecisionTreeClassifierOptions decision_tree_classifier;
        GaussianNBOptions gaussian_nb;
        KNearestNeighborsOptions k_nearest_neighbors;
        SupportVectorMachineOptions support_vector_machine;
        NeuralNetworkOptions neural_network;
        EnsembleModelOptions ensemble_model;
    }BuildClassifierInput;

	typedef structure{
		string attribute_type;
		string attribute;
		float weight;

	}attributeWeights;



	typedef structure{
		string phenotypeclass;
		float accuracy;
		float precision;
		float recall;
		float f1score;
		float dtentropy;
		float dtgini;
		list<attributeWeights> attribute_weights;

	}phenotypeClassInfo;

	typedef structure{
		string classifier_name;
		string classifier_ref;
		list<phenotypeClassInfo> phenotype_class_info;
		float averagef1;
	}classifierInfo;

    typedef structure {
    	list<classifierInfo> classifier_info;
        string report_name;
        string report_ref;
    }ClassifierOut;

    /**
    build_classifier: build_classifier

    requried params:
    **/


    funcdef build_classifier(BuildClassifierInput params)
        returns (ClassifierOut output) authentication required;



   typedef structure {
        string workspace;
        string attribute;
        string classifier_name;
        string description;
        string phenotypeclass;
        string shock_id;
        string list_name;
        int Annotated;
    } ClassifierPredictionInput;


   typedef structure {
        float prediction_accuracy;
        mapping<string genome_id,string predicted_phenotype> predictions;

   }ClassifierPredictionOutput;

   funcdef predict_phenotype(ClassifierPredictionInput params)
        returns (ClassifierPredictionOutput output) authentication required;


	typedef structure {
        string phenotypeclass;
        string workspace;
        string description;
        mapping <string genome_id,ClassifierTrainingSet> classifier_training_set;
        string training_set_out;
        string target;
        string Upload_File;
        int Annotated;
        string list_name;
    }UploadTrainingSetInput;

    typedef structure {
        string phenotype;
        string genome_name;
        string genome_ref;
        int load_status;
        int RAST_annotation_status;
    } ClassifierTrainingSetOut;



    typedef structure {
    	string description;
    	mapping <string genome_id,ClassifierTrainingSetOut> classifier_training_set;
        string report_name;
        string report_ref;
    }UploadTrainingSetOut;


	funcdef upload_trainingset(UploadTrainingSetInput params)
        returns (UploadTrainingSetOut output) authentication required;
};
