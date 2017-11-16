
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
 * <p>Original spec-file type: ClassifierOut</p>
 * 
 * 
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@Generated("com.googlecode.jsonschema2pojo")
@JsonPropertyOrder({
    "classifier_ref",
    "phenotype"
})
public class ClassifierOut {

    @JsonProperty("classifier_ref")
    private String classifierRef;
    @JsonProperty("phenotype")
    private String phenotype;
    private Map<String, Object> additionalProperties = new HashMap<String, Object>();

    @JsonProperty("classifier_ref")
    public String getClassifierRef() {
        return classifierRef;
    }

    @JsonProperty("classifier_ref")
    public void setClassifierRef(String classifierRef) {
        this.classifierRef = classifierRef;
    }

    public ClassifierOut withClassifierRef(String classifierRef) {
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

    public ClassifierOut withPhenotype(String phenotype) {
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
        return ((((((("ClassifierOut"+" [classifierRef=")+ classifierRef)+", phenotype=")+ phenotype)+", additionalProperties=")+ additionalProperties)+"]");
    }

}
