
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
 * <p>Original spec-file type: EnsembleModelOptions</p>
 * 
 * 
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@Generated("com.googlecode.jsonschema2pojo")
@JsonPropertyOrder({
    "k_nearest_neighbors_box",
    "gaussian_nb_box",
    "logistic_regression_box",
    "decision_tree_classifier_box",
    "support_vector_machine_box",
    "neural_network_box",
    "voting",
    "en_weights",
    "en_n_jobs",
    "flatten_transform"
})
public class EnsembleModelOptions {

    @JsonProperty("k_nearest_neighbors_box")
    private Long kNearestNeighborsBox;
    @JsonProperty("gaussian_nb_box")
    private Long gaussianNbBox;
    @JsonProperty("logistic_regression_box")
    private Long logisticRegressionBox;
    @JsonProperty("decision_tree_classifier_box")
    private Long decisionTreeClassifierBox;
    @JsonProperty("support_vector_machine_box")
    private Long supportVectorMachineBox;
    @JsonProperty("neural_network_box")
    private Long neuralNetworkBox;
    @JsonProperty("voting")
    private String voting;
    @JsonProperty("en_weights")
    private String enWeights;
    @JsonProperty("en_n_jobs")
    private Long enNJobs;
    @JsonProperty("flatten_transform")
    private String flattenTransform;
    private Map<String, Object> additionalProperties = new HashMap<String, Object>();

    @JsonProperty("k_nearest_neighbors_box")
    public Long getKNearestNeighborsBox() {
        return kNearestNeighborsBox;
    }

    @JsonProperty("k_nearest_neighbors_box")
    public void setKNearestNeighborsBox(Long kNearestNeighborsBox) {
        this.kNearestNeighborsBox = kNearestNeighborsBox;
    }

    public EnsembleModelOptions withKNearestNeighborsBox(Long kNearestNeighborsBox) {
        this.kNearestNeighborsBox = kNearestNeighborsBox;
        return this;
    }

    @JsonProperty("gaussian_nb_box")
    public Long getGaussianNbBox() {
        return gaussianNbBox;
    }

    @JsonProperty("gaussian_nb_box")
    public void setGaussianNbBox(Long gaussianNbBox) {
        this.gaussianNbBox = gaussianNbBox;
    }

    public EnsembleModelOptions withGaussianNbBox(Long gaussianNbBox) {
        this.gaussianNbBox = gaussianNbBox;
        return this;
    }

    @JsonProperty("logistic_regression_box")
    public Long getLogisticRegressionBox() {
        return logisticRegressionBox;
    }

    @JsonProperty("logistic_regression_box")
    public void setLogisticRegressionBox(Long logisticRegressionBox) {
        this.logisticRegressionBox = logisticRegressionBox;
    }

    public EnsembleModelOptions withLogisticRegressionBox(Long logisticRegressionBox) {
        this.logisticRegressionBox = logisticRegressionBox;
        return this;
    }

    @JsonProperty("decision_tree_classifier_box")
    public Long getDecisionTreeClassifierBox() {
        return decisionTreeClassifierBox;
    }

    @JsonProperty("decision_tree_classifier_box")
    public void setDecisionTreeClassifierBox(Long decisionTreeClassifierBox) {
        this.decisionTreeClassifierBox = decisionTreeClassifierBox;
    }

    public EnsembleModelOptions withDecisionTreeClassifierBox(Long decisionTreeClassifierBox) {
        this.decisionTreeClassifierBox = decisionTreeClassifierBox;
        return this;
    }

    @JsonProperty("support_vector_machine_box")
    public Long getSupportVectorMachineBox() {
        return supportVectorMachineBox;
    }

    @JsonProperty("support_vector_machine_box")
    public void setSupportVectorMachineBox(Long supportVectorMachineBox) {
        this.supportVectorMachineBox = supportVectorMachineBox;
    }

    public EnsembleModelOptions withSupportVectorMachineBox(Long supportVectorMachineBox) {
        this.supportVectorMachineBox = supportVectorMachineBox;
        return this;
    }

    @JsonProperty("neural_network_box")
    public Long getNeuralNetworkBox() {
        return neuralNetworkBox;
    }

    @JsonProperty("neural_network_box")
    public void setNeuralNetworkBox(Long neuralNetworkBox) {
        this.neuralNetworkBox = neuralNetworkBox;
    }

    public EnsembleModelOptions withNeuralNetworkBox(Long neuralNetworkBox) {
        this.neuralNetworkBox = neuralNetworkBox;
        return this;
    }

    @JsonProperty("voting")
    public String getVoting() {
        return voting;
    }

    @JsonProperty("voting")
    public void setVoting(String voting) {
        this.voting = voting;
    }

    public EnsembleModelOptions withVoting(String voting) {
        this.voting = voting;
        return this;
    }

    @JsonProperty("en_weights")
    public String getEnWeights() {
        return enWeights;
    }

    @JsonProperty("en_weights")
    public void setEnWeights(String enWeights) {
        this.enWeights = enWeights;
    }

    public EnsembleModelOptions withEnWeights(String enWeights) {
        this.enWeights = enWeights;
        return this;
    }

    @JsonProperty("en_n_jobs")
    public Long getEnNJobs() {
        return enNJobs;
    }

    @JsonProperty("en_n_jobs")
    public void setEnNJobs(Long enNJobs) {
        this.enNJobs = enNJobs;
    }

    public EnsembleModelOptions withEnNJobs(Long enNJobs) {
        this.enNJobs = enNJobs;
        return this;
    }

    @JsonProperty("flatten_transform")
    public String getFlattenTransform() {
        return flattenTransform;
    }

    @JsonProperty("flatten_transform")
    public void setFlattenTransform(String flattenTransform) {
        this.flattenTransform = flattenTransform;
    }

    public EnsembleModelOptions withFlattenTransform(String flattenTransform) {
        this.flattenTransform = flattenTransform;
        return this;
    }

    @JsonAnyGetter
    public Map<String, Object> getAdditionalProperties() {
        return this.additionalProperties;
    }

    @JsonAnySetter
    public void setAdditionalProperties(String name, Object value) {
        this.additionalProperties.put(name, value);
    }

    @Override
    public String toString() {
        return ((((((((((((((((((((((("EnsembleModelOptions"+" [kNearestNeighborsBox=")+ kNearestNeighborsBox)+", gaussianNbBox=")+ gaussianNbBox)+", logisticRegressionBox=")+ logisticRegressionBox)+", decisionTreeClassifierBox=")+ decisionTreeClassifierBox)+", supportVectorMachineBox=")+ supportVectorMachineBox)+", neuralNetworkBox=")+ neuralNetworkBox)+", voting=")+ voting)+", enWeights=")+ enWeights)+", enNJobs=")+ enNJobs)+", flattenTransform=")+ flattenTransform)+", additionalProperties=")+ additionalProperties)+"]");
    }

}
