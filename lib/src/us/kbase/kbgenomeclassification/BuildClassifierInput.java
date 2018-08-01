
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
    "classifier_training_set",
    "classifier_out",
    "target",
    "classifier",
    "shock_id",
    "list_name",
    "save_ts"
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
    @JsonProperty("classifier_out")
    private java.lang.String classifierOut;
    @JsonProperty("target")
    private java.lang.String target;
    @JsonProperty("classifier")
    private java.lang.String classifier;
    @JsonProperty("shock_id")
    private java.lang.String shockId;
    @JsonProperty("list_name")
    private java.lang.String listName;
    @JsonProperty("save_ts")
    private Long saveTs;
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

    @JsonProperty("classifier_out")
    public java.lang.String getClassifierOut() {
        return classifierOut;
    }

    @JsonProperty("classifier_out")
    public void setClassifierOut(java.lang.String classifierOut) {
        this.classifierOut = classifierOut;
    }

    public BuildClassifierInput withClassifierOut(java.lang.String classifierOut) {
        this.classifierOut = classifierOut;
        return this;
    }

    @JsonProperty("target")
    public java.lang.String getTarget() {
        return target;
    }

    @JsonProperty("target")
    public void setTarget(java.lang.String target) {
        this.target = target;
    }

    public BuildClassifierInput withTarget(java.lang.String target) {
        this.target = target;
        return this;
    }

    @JsonProperty("classifier")
    public java.lang.String getClassifier() {
        return classifier;
    }

    @JsonProperty("classifier")
    public void setClassifier(java.lang.String classifier) {
        this.classifier = classifier;
    }

    public BuildClassifierInput withClassifier(java.lang.String classifier) {
        this.classifier = classifier;
        return this;
    }

    @JsonProperty("shock_id")
    public java.lang.String getShockId() {
        return shockId;
    }

    @JsonProperty("shock_id")
    public void setShockId(java.lang.String shockId) {
        this.shockId = shockId;
    }

    public BuildClassifierInput withShockId(java.lang.String shockId) {
        this.shockId = shockId;
        return this;
    }

    @JsonProperty("list_name")
    public java.lang.String getListName() {
        return listName;
    }

    @JsonProperty("list_name")
    public void setListName(java.lang.String listName) {
        this.listName = listName;
    }

    public BuildClassifierInput withListName(java.lang.String listName) {
        this.listName = listName;
        return this;
    }

    @JsonProperty("save_ts")
    public Long getSaveTs() {
        return saveTs;
    }

    @JsonProperty("save_ts")
    public void setSaveTs(Long saveTs) {
        this.saveTs = saveTs;
    }

    public BuildClassifierInput withSaveTs(Long saveTs) {
        this.saveTs = saveTs;
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
        return ((((((((((((((((((((((("BuildClassifierInput"+" [phenotypeclass=")+ phenotypeclass)+", attribute=")+ attribute)+", workspace=")+ workspace)+", classifierTrainingSet=")+ classifierTrainingSet)+", classifierOut=")+ classifierOut)+", target=")+ target)+", classifier=")+ classifier)+", shockId=")+ shockId)+", listName=")+ listName)+", saveTs=")+ saveTs)+", additionalProperties=")+ additionalProperties)+"]");
    }

}
