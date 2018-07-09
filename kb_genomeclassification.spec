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
    }BuildClassifierInput;

    typedef structure {
       string classifier_ref;

       string phenotype;
    }ClassifierOut;

    /**
    build_classifier: build_classifier

    requried params:

    ss
    **/
    funcdef build_classifier(BuildClassifierInput params)
        returns (ClassifierOut output) authentication required;



   typedef structure {
        string workspace;
        string classifier_ref;
        string phenotype;
    } ClassifierPredictionInput;


   typedef structure {
        float prediction_accuracy;
        mapping<string genome_id,string predicted_phenotype> predictions;

   }ClassifierPredictionOutput;

   funcdef predict_phenotype(ClassifierPredictionInput params)
        returns (ClassifierPredictionOutput output) authentication required;

};
