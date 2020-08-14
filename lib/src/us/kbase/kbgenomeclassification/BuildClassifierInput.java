
package us.kbase.kbgenomeclassification;

import java.util.HashMap;
import java.util.Map;
import javax.annotation.Generated;
import com.fasterxml.jackson.annotation.JsonAnyGetter;
import com.fasterxml.jackson.annotation.JsonAnySetter;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;


/**
 * <p>Original spec-file type: BuildClassifierInput</p>
 * 
 * 
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@Generated("com.googlecode.jsonschema2pojo")
@JsonPropertyOrder({
    "genome_attribute",
    "workspace",
    "training_set_name",
    "classifier_training_set",
    "classifier_object_name",
    "description",
    "classifier_to_run",
    "logistic_regression",
    "decision_tree_classifier",
    "gaussian_nb",
    "k_nearest_neighbors",
    "support_vector_machine",
    "neural_network",
    "ensemble_model"
})
public class BuildClassifierInput {

    @JsonProperty("genome_attribute")
    private java.lang.String genomeAttribute;
    @JsonProperty("workspace")
    private java.lang.String workspace;
    @JsonProperty("training_set_name")
    private java.lang.String trainingSetName;
    @JsonProperty("classifier_training_set")
    private Map<String, ClassifierTrainingSet> classifierTrainingSet;
    @JsonProperty("classifier_object_name")
    private java.lang.String classifierObjectName;
    @JsonProperty("description")
    private java.lang.String description;
    @JsonProperty("classifier_to_run")
    private java.lang.String classifierToRun;
    /**
     * <p>Original spec-file type: LogisticRegressionOptions</p>
     * 
     * 
     */
    @JsonProperty("logistic_regression")
    private LogisticRegressionOptions logisticRegression;
    /**
     * <p>Original spec-file type: DecisionTreeClassifierOptions</p>
     * 
     * 
     */
    @JsonProperty("decision_tree_classifier")
    private DecisionTreeClassifierOptions decisionTreeClassifier;
    /**
     * <p>Original spec-file type: GaussianNBOptions</p>
     * 
     * 
     */
    @JsonProperty("gaussian_nb")
    private GaussianNBOptions gaussianNb;
    /**
     * <p>Original spec-file type: KNearestNeighborsOptions</p>
     * 
     * 
     */
    @JsonProperty("k_nearest_neighbors")
    private KNearestNeighborsOptions kNearestNeighbors;
    /**
     * <p>Original spec-file type: SupportVectorMachineOptions</p>
     * 
     * 
     */
    @JsonProperty("support_vector_machine")
    private SupportVectorMachineOptions supportVectorMachine;
    /**
     * <p>Original spec-file type: NeuralNetworkOptions</p>
     * 
     * 
     */
    @JsonProperty("neural_network")
    private NeuralNetworkOptions neuralNetwork;
    /**
     * <p>Original spec-file type: EnsembleModelOptions</p>
     * 
     * 
     */
    @JsonProperty("ensemble_model")
    private EnsembleModelOptions ensembleModel;
    private Map<java.lang.String, Object> additionalProperties = new HashMap<java.lang.String, Object>();

    @JsonProperty("genome_attribute")
    public java.lang.String getGenomeAttribute() {
        return genomeAttribute;
    }

    @JsonProperty("genome_attribute")
    public void setGenomeAttribute(java.lang.String genomeAttribute) {
        this.genomeAttribute = genomeAttribute;
    }

    public BuildClassifierInput withGenomeAttribute(java.lang.String genomeAttribute) {
        this.genomeAttribute = genomeAttribute;
        return this;
    }

    @JsonProperty("workspace")
    public java.lang.String getWorkspace() {
        return workspace;
    }

    @JsonProperty("workspace")
    public void setWorkspace(java.lang.String workspace) {
        this.workspace = workspace;
    }

    public BuildClassifierInput withWorkspace(java.lang.String workspace) {
        this.workspace = workspace;
        return this;
    }

    @JsonProperty("training_set_name")
    public java.lang.String getTrainingSetName() {
        return trainingSetName;
    }

    @JsonProperty("training_set_name")
    public void setTrainingSetName(java.lang.String trainingSetName) {
        this.trainingSetName = trainingSetName;
    }

    public BuildClassifierInput withTrainingSetName(java.lang.String trainingSetName) {
        this.trainingSetName = trainingSetName;
        return this;
    }

    @JsonProperty("classifier_training_set")
    public Map<String, ClassifierTrainingSet> getClassifierTrainingSet() {
        return classifierTrainingSet;
    }

    @JsonProperty("classifier_training_set")
    public void setClassifierTrainingSet(Map<String, ClassifierTrainingSet> classifierTrainingSet) {
        this.classifierTrainingSet = classifierTrainingSet;
    }

    public BuildClassifierInput withClassifierTrainingSet(Map<String, ClassifierTrainingSet> classifierTrainingSet) {
        this.classifierTrainingSet = classifierTrainingSet;
        return this;
    }

    @JsonProperty("classifier_object_name")
    public java.lang.String getClassifierObjectName() {
        return classifierObjectName;
    }

    @JsonProperty("classifier_object_name")
    public void setClassifierObjectName(java.lang.String classifierObjectName) {
        this.classifierObjectName = classifierObjectName;
    }

    public BuildClassifierInput withClassifierObjectName(java.lang.String classifierObjectName) {
        this.classifierObjectName = classifierObjectName;
        return this;
    }

    @JsonProperty("description")
    public java.lang.String getDescription() {
        return description;
    }

    @JsonProperty("description")
    public void setDescription(java.lang.String description) {
        this.description = description;
    }

    public BuildClassifierInput withDescription(java.lang.String description) {
        this.description = description;
        return this;
    }

    @JsonProperty("classifier_to_run")
    public java.lang.String getClassifierToRun() {
        return classifierToRun;
    }

    @JsonProperty("classifier_to_run")
    public void setClassifierToRun(java.lang.String classifierToRun) {
        this.classifierToRun = classifierToRun;
    }

    public BuildClassifierInput withClassifierToRun(java.lang.String classifierToRun) {
        this.classifierToRun = classifierToRun;
        return this;
    }

    /**
     * <p>Original spec-file type: LogisticRegressionOptions</p>
     * 
     * 
     */
    @JsonProperty("logistic_regression")
    public LogisticRegressionOptions getLogisticRegression() {
        return logisticRegression;
    }

    /**
     * <p>Original spec-file type: LogisticRegressionOptions</p>
     * 
     * 
     */
    @JsonProperty("logistic_regression")
    public void setLogisticRegression(LogisticRegressionOptions logisticRegression) {
        this.logisticRegression = logisticRegression;
    }

    public BuildClassifierInput withLogisticRegression(LogisticRegressionOptions logisticRegression) {
        this.logisticRegression = logisticRegression;
        return this;
    }

    /**
     * <p>Original spec-file type: DecisionTreeClassifierOptions</p>
     * 
     * 
     */
    @JsonProperty("decision_tree_classifier")
    public DecisionTreeClassifierOptions getDecisionTreeClassifier() {
        return decisionTreeClassifier;
    }

    /**
     * <p>Original spec-file type: DecisionTreeClassifierOptions</p>
     * 
     * 
     */
    @JsonProperty("decision_tree_classifier")
    public void setDecisionTreeClassifier(DecisionTreeClassifierOptions decisionTreeClassifier) {
        this.decisionTreeClassifier = decisionTreeClassifier;
    }

    public BuildClassifierInput withDecisionTreeClassifier(DecisionTreeClassifierOptions decisionTreeClassifier) {
        this.decisionTreeClassifier = decisionTreeClassifier;
        return this;
    }

    /**
     * <p>Original spec-file type: GaussianNBOptions</p>
     * 
     * 
     */
    @JsonProperty("gaussian_nb")
    public GaussianNBOptions getGaussianNb() {
        return gaussianNb;
    }

    /**
     * <p>Original spec-file type: GaussianNBOptions</p>
     * 
     * 
     */
    @JsonProperty("gaussian_nb")
    public void setGaussianNb(GaussianNBOptions gaussianNb) {
        this.gaussianNb = gaussianNb;
    }

    public BuildClassifierInput withGaussianNb(GaussianNBOptions gaussianNb) {
        this.gaussianNb = gaussianNb;
        return this;
    }

    /**
     * <p>Original spec-file type: KNearestNeighborsOptions</p>
     * 
     * 
     */
    @JsonProperty("k_nearest_neighbors")
    public KNearestNeighborsOptions getKNearestNeighbors() {
        return kNearestNeighbors;
    }

    /**
     * <p>Original spec-file type: KNearestNeighborsOptions</p>
     * 
     * 
     */
    @JsonProperty("k_nearest_neighbors")
    public void setKNearestNeighbors(KNearestNeighborsOptions kNearestNeighbors) {
        this.kNearestNeighbors = kNearestNeighbors;
    }

    public BuildClassifierInput withKNearestNeighbors(KNearestNeighborsOptions kNearestNeighbors) {
        this.kNearestNeighbors = kNearestNeighbors;
        return this;
    }

    /**
     * <p>Original spec-file type: SupportVectorMachineOptions</p>
     * 
     * 
     */
    @JsonProperty("support_vector_machine")
    public SupportVectorMachineOptions getSupportVectorMachine() {
        return supportVectorMachine;
    }

    /**
     * <p>Original spec-file type: SupportVectorMachineOptions</p>
     * 
     * 
     */
    @JsonProperty("support_vector_machine")
    public void setSupportVectorMachine(SupportVectorMachineOptions supportVectorMachine) {
        this.supportVectorMachine = supportVectorMachine;
    }

    public BuildClassifierInput withSupportVectorMachine(SupportVectorMachineOptions supportVectorMachine) {
        this.supportVectorMachine = supportVectorMachine;
        return this;
    }

    /**
     * <p>Original spec-file type: NeuralNetworkOptions</p>
     * 
     * 
     */
    @JsonProperty("neural_network")
    public NeuralNetworkOptions getNeuralNetwork() {
        return neuralNetwork;
    }

    /**
     * <p>Original spec-file type: NeuralNetworkOptions</p>
     * 
     * 
     */
    @JsonProperty("neural_network")
    public void setNeuralNetwork(NeuralNetworkOptions neuralNetwork) {
        this.neuralNetwork = neuralNetwork;
    }

    public BuildClassifierInput withNeuralNetwork(NeuralNetworkOptions neuralNetwork) {
        this.neuralNetwork = neuralNetwork;
        return this;
    }

    /**
     * <p>Original spec-file type: EnsembleModelOptions</p>
     * 
     * 
     */
    @JsonProperty("ensemble_model")
    public EnsembleModelOptions getEnsembleModel() {
        return ensembleModel;
    }

    /**
     * <p>Original spec-file type: EnsembleModelOptions</p>
     * 
     * 
     */
    @JsonProperty("ensemble_model")
    public void setEnsembleModel(EnsembleModelOptions ensembleModel) {
        this.ensembleModel = ensembleModel;
    }

    public BuildClassifierInput withEnsembleModel(EnsembleModelOptions ensembleModel) {
        this.ensembleModel = ensembleModel;
        return this;
    }

    @JsonAnyGetter
    public Map<java.lang.String, Object> getAdditionalProperties() {
        return this.additionalProperties;
    }

    @JsonAnySetter
    public void setAdditionalProperties(java.lang.String name, Object value) {
        this.additionalProperties.put(name, value);
    }

    @Override
    public java.lang.String toString() {
        return ((((((((((((((((((((((((((((((("BuildClassifierInput"+" [genomeAttribute=")+ genomeAttribute)+", workspace=")+ workspace)+", trainingSetName=")+ trainingSetName)+", classifierTrainingSet=")+ classifierTrainingSet)+", classifierObjectName=")+ classifierObjectName)+", description=")+ description)+", classifierToRun=")+ classifierToRun)+", logisticRegression=")+ logisticRegression)+", decisionTreeClassifier=")+ decisionTreeClassifier)+", gaussianNb=")+ gaussianNb)+", kNearestNeighbors=")+ kNearestNeighbors)+", supportVectorMachine=")+ supportVectorMachine)+", neuralNetwork=")+ neuralNetwork)+", ensembleModel=")+ ensembleModel)+", additionalProperties=")+ additionalProperties)+"]");
    }

}
