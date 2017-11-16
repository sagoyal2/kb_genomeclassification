
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
 * <p>Original spec-file type: ClassifierPredictionInput</p>
 * 
 * 
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@Generated("com.googlecode.jsonschema2pojo")
@JsonPropertyOrder({
    "workspace",
    "classifier_ref",
    "phenotype"
})
public class ClassifierPredictionInput {

    @JsonProperty("workspace")
    private String workspace;
    @JsonProperty("classifier_ref")
    private String classifierRef;
    @JsonProperty("phenotype")
    private String phenotype;
    private Map<String, Object> additionalProperties = new HashMap<String, Object>();

    @JsonProperty("workspace")
    public String getWorkspace() {
        return workspace;
    }

    @JsonProperty("workspace")
    public void setWorkspace(String workspace) {
        this.workspace = workspace;
    }

    public ClassifierPredictionInput withWorkspace(String workspace) {
        this.workspace = workspace;
        return this;
    }

    @JsonProperty("classifier_ref")
    public String getClassifierRef() {
        return classifierRef;
    }

    @JsonProperty("classifier_ref")
    public void setClassifierRef(String classifierRef) {
        this.classifierRef = classifierRef;
    }

    public ClassifierPredictionInput withClassifierRef(String classifierRef) {
        this.classifierRef = classifierRef;
        return this;
    }

    @JsonProperty("phenotype")
    public String getPhenotype() {
        return phenotype;
    }

    @JsonProperty("phenotype")
    public void setPhenotype(String phenotype) {
        this.phenotype = phenotype;
    }

    public ClassifierPredictionInput withPhenotype(String phenotype) {
        this.phenotype = phenotype;
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
        return ((((((((("ClassifierPredictionInput"+" [workspace=")+ workspace)+", classifierRef=")+ classifierRef)+", phenotype=")+ phenotype)+", additionalProperties=")+ additionalProperties)+"]");
    }

}
