
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
    "classifier_name",
    "phenotypeclass",
    "shock_id",
    "list_name"
})
public class ClassifierPredictionInput {

    @JsonProperty("workspace")
    private String workspace;
    @JsonProperty("classifier_name")
    private String classifierName;
    @JsonProperty("phenotypeclass")
    private String phenotypeclass;
    @JsonProperty("shock_id")
    private String shockId;
    @JsonProperty("list_name")
    private String listName;
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

    @JsonProperty("classifier_name")
    public String getClassifierName() {
        return classifierName;
    }

    @JsonProperty("classifier_name")
    public void setClassifierName(String classifierName) {
        this.classifierName = classifierName;
    }

    public ClassifierPredictionInput withClassifierName(String classifierName) {
        this.classifierName = classifierName;
        return this;
    }

    @JsonProperty("phenotypeclass")
    public String getPhenotypeclass() {
        return phenotypeclass;
    }

    @JsonProperty("phenotypeclass")
    public void setPhenotypeclass(String phenotypeclass) {
        this.phenotypeclass = phenotypeclass;
    }

    public ClassifierPredictionInput withPhenotypeclass(String phenotypeclass) {
        this.phenotypeclass = phenotypeclass;
        return this;
    }

    @JsonProperty("shock_id")
    public String getShockId() {
        return shockId;
    }

    @JsonProperty("shock_id")
    public void setShockId(String shockId) {
        this.shockId = shockId;
    }

    public ClassifierPredictionInput withShockId(String shockId) {
        this.shockId = shockId;
        return this;
    }

    @JsonProperty("list_name")
    public String getListName() {
        return listName;
    }

    @JsonProperty("list_name")
    public void setListName(String listName) {
        this.listName = listName;
    }

    public ClassifierPredictionInput withListName(String listName) {
        this.listName = listName;
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
        return ((((((((((((("ClassifierPredictionInput"+" [workspace=")+ workspace)+", classifierName=")+ classifierName)+", phenotypeclass=")+ phenotypeclass)+", shockId=")+ shockId)+", listName=")+ listName)+", additionalProperties=")+ additionalProperties)+"]");
    }

}
