
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
 * <p>Original spec-file type: UploadTrainingSetInput</p>
 * 
 * 
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@Generated("com.googlecode.jsonschema2pojo")
@JsonPropertyOrder({
    "phenotype",
    "workspace",
    "workspace_id",
    "description",
    "training_set_name",
    "file_path",
    "annotate"
})
public class UploadTrainingSetInput {

    @JsonProperty("phenotype")
    private String phenotype;
    @JsonProperty("workspace")
    private String workspace;
    @JsonProperty("workspace_id")
    private String workspaceId;
    @JsonProperty("description")
    private String description;
    @JsonProperty("training_set_name")
    private String trainingSetName;
    @JsonProperty("file_path")
    private String filePath;
    @JsonProperty("annotate")
    private Long annotate;
    private Map<String, Object> additionalProperties = new HashMap<String, Object>();

    @JsonProperty("phenotype")
    public String getPhenotype() {
        return phenotype;
    }

    @JsonProperty("phenotype")
    public void setPhenotype(String phenotype) {
        this.phenotype = phenotype;
    }

    public UploadTrainingSetInput withPhenotype(String phenotype) {
        this.phenotype = phenotype;
        return this;
    }

    @JsonProperty("workspace")
    public String getWorkspace() {
        return workspace;
    }

    @JsonProperty("workspace")
    public void setWorkspace(String workspace) {
        this.workspace = workspace;
    }

    public UploadTrainingSetInput withWorkspace(String workspace) {
        this.workspace = workspace;
        return this;
    }

    @JsonProperty("workspace_id")
    public String getWorkspaceId() {
        return workspaceId;
    }

    @JsonProperty("workspace_id")
    public void setWorkspaceId(String workspaceId) {
        this.workspaceId = workspaceId;
    }

    public UploadTrainingSetInput withWorkspaceId(String workspaceId) {
        this.workspaceId = workspaceId;
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

    public UploadTrainingSetInput withDescription(String description) {
        this.description = description;
        return this;
    }

    @JsonProperty("training_set_name")
    public String getTrainingSetName() {
        return trainingSetName;
    }

    @JsonProperty("training_set_name")
    public void setTrainingSetName(String trainingSetName) {
        this.trainingSetName = trainingSetName;
    }

    public UploadTrainingSetInput withTrainingSetName(String trainingSetName) {
        this.trainingSetName = trainingSetName;
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

    public UploadTrainingSetInput withFilePath(String filePath) {
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

    public UploadTrainingSetInput withAnnotate(Long annotate) {
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
        return ((((((((((((((((("UploadTrainingSetInput"+" [phenotype=")+ phenotype)+", workspace=")+ workspace)+", workspaceId=")+ workspaceId)+", description=")+ description)+", trainingSetName=")+ trainingSetName)+", filePath=")+ filePath)+", annotate=")+ annotate)+", additionalProperties=")+ additionalProperties)+"]");
    }

}
