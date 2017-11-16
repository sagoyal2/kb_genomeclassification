
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
    "phenotypeclass",
    "attribute",
    "workspace",
    "classifier_training_set"
})
public class BuildClassifierInput {

    @JsonProperty("phenotypeclass")
    private java.lang.String phenotypeclass;
    @JsonProperty("attribute")
    private java.lang.String attribute;
    @JsonProperty("workspace")
    private java.lang.String workspace;
    @JsonProperty("classifier_training_set")
    private Map<String, ClassifierTrainingSet> classifierTrainingSet;
    private Map<java.lang.String, Object> additionalProperties = new HashMap<java.lang.String, Object>();

    @JsonProperty("phenotypeclass")
    public java.lang.String getPhenotypeclass() {
        return phenotypeclass;
    }

    @JsonProperty("phenotypeclass")
    public void setPhenotypeclass(java.lang.String phenotypeclass) {
        this.phenotypeclass = phenotypeclass;
    }

    public BuildClassifierInput withPhenotypeclass(java.lang.String phenotypeclass) {
        this.phenotypeclass = phenotypeclass;
        return this;
    }

    @JsonProperty("attribute")
    public java.lang.String getAttribute() {
        return attribute;
    }

    @JsonProperty("attribute")
    public void setAttribute(java.lang.String attribute) {
        this.attribute = attribute;
    }

    public BuildClassifierInput withAttribute(java.lang.String attribute) {
        this.attribute = attribute;
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
        return ((((((((((("BuildClassifierInput"+" [phenotypeclass=")+ phenotypeclass)+", attribute=")+ attribute)+", workspace=")+ workspace)+", classifierTrainingSet=")+ classifierTrainingSet)+", additionalProperties=")+ additionalProperties)+"]");
    }

}
