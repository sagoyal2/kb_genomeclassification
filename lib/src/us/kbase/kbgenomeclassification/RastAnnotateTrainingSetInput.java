
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
 * <p>Original spec-file type: RastAnnotateTrainingSetInput</p>
 * 
 * 
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@Generated("com.googlecode.jsonschema2pojo")
@JsonPropertyOrder({
    "classifier_training_set",
    "workspace",
    "make_genome_set"
})
public class RastAnnotateTrainingSetInput {

    @JsonProperty("classifier_training_set")
    private Map<String, ClassifierTrainingSetOut> classifierTrainingSet;
    @JsonProperty("workspace")
    private java.lang.String workspace;
    @JsonProperty("make_genome_set")
    private Long makeGenomeSet;
    private Map<java.lang.String, Object> additionalProperties = new HashMap<java.lang.String, Object>();

    @JsonProperty("classifier_training_set")
    public Map<String, ClassifierTrainingSetOut> getClassifierTrainingSet() {
        return classifierTrainingSet;
    }

    @JsonProperty("classifier_training_set")
    public void setClassifierTrainingSet(Map<String, ClassifierTrainingSetOut> classifierTrainingSet) {
        this.classifierTrainingSet = classifierTrainingSet;
    }

    public RastAnnotateTrainingSetInput withClassifierTrainingSet(Map<String, ClassifierTrainingSetOut> classifierTrainingSet) {
        this.classifierTrainingSet = classifierTrainingSet;
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

    public RastAnnotateTrainingSetInput withWorkspace(java.lang.String workspace) {
        this.workspace = workspace;
        return this;
    }

    @JsonProperty("make_genome_set")
    public Long getMakeGenomeSet() {
        return makeGenomeSet;
    }

    @JsonProperty("make_genome_set")
    public void setMakeGenomeSet(Long makeGenomeSet) {
        this.makeGenomeSet = makeGenomeSet;
    }

    public RastAnnotateTrainingSetInput withMakeGenomeSet(Long makeGenomeSet) {
        this.makeGenomeSet = makeGenomeSet;
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
        return ((((((((("RastAnnotateTrainingSetInput"+" [classifierTrainingSet=")+ classifierTrainingSet)+", workspace=")+ workspace)+", makeGenomeSet=")+ makeGenomeSet)+", additionalProperties=")+ additionalProperties)+"]");
    }

}
