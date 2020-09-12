
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
    "categorizer_name",
    "description",
    "file_path",
    "annotate"
})
public class ClassifierPredictionInput {

    @JsonProperty("workspace")
    private String workspace;
    @JsonProperty("categorizer_name")
    private String categorizerName;
    @JsonProperty("description")
    private String description;
    @JsonProperty("file_path")
    private String filePath;
    @JsonProperty("annotate")
    private Long annotate;
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

    @JsonProperty("categorizer_name")
    public String getCategorizerName() {
        return categorizerName;
    }

    @JsonProperty("categorizer_name")
    public void setCategorizerName(String categorizerName) {
        this.categorizerName = categorizerName;
    }

    public ClassifierPredictionInput withCategorizerName(String categorizerName) {
        this.categorizerName = categorizerName;
        return this;
    }

    @JsonProperty("description")
    public String getDescription() {
        return description;
    }

    @JsonProperty("description")
    public void setDescription(String description) {
        this.description = description;
    }

    public ClassifierPredictionInput withDescription(String description) {
        this.description = description;
        return this;
    }

    @JsonProperty("file_path")
    public String getFilePath() {
        return filePath;
    }

    @JsonProperty("file_path")
    public void setFilePath(String filePath) {
        this.filePath = filePath;
    }

    public ClassifierPredictionInput withFilePath(String filePath) {
        this.filePath = filePath;
        return this;
    }

    @JsonProperty("annotate")
    public Long getAnnotate() {
        return annotate;
    }

    @JsonProperty("annotate")
    public void setAnnotate(Long annotate) {
        this.annotate = annotate;
    }

    public ClassifierPredictionInput withAnnotate(Long annotate) {
        this.annotate = annotate;
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
        return ((((((((((((("ClassifierPredictionInput"+" [workspace=")+ workspace)+", categorizerName=")+ categorizerName)+", description=")+ description)+", filePath=")+ filePath)+", annotate=")+ annotate)+", additionalProperties=")+ additionalProperties)+"]");
    }

}
