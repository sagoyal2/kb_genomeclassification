
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
 * <p>Original spec-file type: UploadTrainingSetOut</p>
 * 
 * 
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@Generated("com.googlecode.jsonschema2pojo")
@JsonPropertyOrder({
    "classifier_training_set",
    "report_name",
    "report_ref"
})
public class UploadTrainingSetOut {

    @JsonProperty("classifier_training_set")
    private Map<String, ClassifierTrainingSetOut> classifierTrainingSet;
    @JsonProperty("report_name")
    private java.lang.String reportName;
    @JsonProperty("report_ref")
    private java.lang.String reportRef;
    private Map<java.lang.String, Object> additionalProperties = new HashMap<java.lang.String, Object>();

    @JsonProperty("classifier_training_set")
    public Map<String, ClassifierTrainingSetOut> getClassifierTrainingSet() {
        return classifierTrainingSet;
    }

    @JsonProperty("classifier_training_set")
    public void setClassifierTrainingSet(Map<String, ClassifierTrainingSetOut> classifierTrainingSet) {
        this.classifierTrainingSet = classifierTrainingSet;
    }

    public UploadTrainingSetOut withClassifierTrainingSet(Map<String, ClassifierTrainingSetOut> classifierTrainingSet) {
        this.classifierTrainingSet = classifierTrainingSet;
        return this;
    }

    @JsonProperty("report_name")
    public java.lang.String getReportName() {
        return reportName;
    }

    @JsonProperty("report_name")
    public void setReportName(java.lang.String reportName) {
        this.reportName = reportName;
    }

    public UploadTrainingSetOut withReportName(java.lang.String reportName) {
        this.reportName = reportName;
        return this;
    }

    @JsonProperty("report_ref")
    public java.lang.String getReportRef() {
        return reportRef;
    }

    @JsonProperty("report_ref")
    public void setReportRef(java.lang.String reportRef) {
        this.reportRef = reportRef;
    }

    public UploadTrainingSetOut withReportRef(java.lang.String reportRef) {
        this.reportRef = reportRef;
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
        return ((((((((("UploadTrainingSetOut"+" [classifierTrainingSet=")+ classifierTrainingSet)+", reportName=")+ reportName)+", reportRef=")+ reportRef)+", additionalProperties=")+ additionalProperties)+"]");
    }

}
