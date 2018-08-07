/*
A KBase module: kb_genomeclassification
This module build a classifier and predict phenotypes based on the classifier
*/

module kb_genomeclassification {

    /* typedef string genome_id;
       typedef string phenotype; */

    typedef structure {
        string phenotype;
        string genome_name;
    } ClassifierTrainingSet;


    typedef structure {
        string phenotypeclass;
        string attribute;
        string workspace;
        mapping <string genome_id,ClassifierTrainingSet> classifier_training_set;
        string classifier_out;
        string target;
        string classifier;
        string shock_id;
        string list_name;
        int save_ts;
    }BuildClassifierInput;

    typedef structure {
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
        string classifier_name;
        string phenotypeclass;
        string shock_id;
        string list_name;
    } ClassifierPredictionInput;


   typedef structure {
        float prediction_accuracy;
        mapping<string genome_id,string predicted_phenotype> predictions;

   }ClassifierPredictionOutput;

   funcdef predict_phenotype(ClassifierPredictionInput params)
        returns (ClassifierPredictionOutput output) authentication required;

};
