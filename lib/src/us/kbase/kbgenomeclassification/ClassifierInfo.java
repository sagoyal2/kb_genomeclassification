
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
 * <p>Original spec-file type: classifierInfo</p>
 * 
 * 
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@Generated("com.googlecode.jsonschema2pojo")
@JsonPropertyOrder({
    "classifier_name",
    "classifier_ref",
    "accuracy"
})
public class ClassifierInfo {

    @JsonProperty("classifier_name")
    private String classifierName;
    @JsonProperty("classifier_ref")
    private String classifierRef;
    @JsonProperty("accuracy")
    private Double accuracy;
    private Map<String, Object> additionalProperties = new HashMap<String, Object>();

    @JsonProperty("classifier_name")
    public String getClassifierName() {
        return classifierName;
    }

    @JsonProperty("classifier_name")
    public void setClassifierName(String classifierName) {
        this.classifierName = classifierName;
    }

    public ClassifierInfo withClassifierName(String classifierName) {
        this.classifierName = classifierName;
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

    public ClassifierInfo withClassifierRef(String classifierRef) {
        this.classifierRef = classifierRef;
        return this;
    }

    @JsonProperty("accuracy")
    public Double getAccuracy() {
        return accuracy;
    }

    @JsonProperty("accuracy")
    public void setAccuracy(Double accuracy) {
        this.accuracy = accuracy;
    }

    public ClassifierInfo withAccuracy(Double accuracy) {
        this.accuracy = accuracy;
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
        return ((((((((("ClassifierInfo"+" [classifierName=")+ classifierName)+", classifierRef=")+ classifierRef)+", accuracy=")+ accuracy)+", additionalProperties=")+ additionalProperties)+"]");
    }

}
