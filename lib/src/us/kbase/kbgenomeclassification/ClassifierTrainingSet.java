
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
 * <p>Original spec-file type: ClassifierTrainingSet</p>
 * <pre>
 * typedef string genome_id;
 * typedef string phenotype;
 * </pre>
 * 
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@Generated("com.googlecode.jsonschema2pojo")
@JsonPropertyOrder({
    "phenotype",
    "genome_name"
})
public class ClassifierTrainingSet {

    @JsonProperty("phenotype")
    private String phenotype;
    @JsonProperty("genome_name")
    private String genomeName;
    private Map<String, Object> additionalProperties = new HashMap<String, Object>();

    @JsonProperty("phenotype")
    public String getPhenotype() {
        return phenotype;
    }

    @JsonProperty("phenotype")
    public void setPhenotype(String phenotype) {
        this.phenotype = phenotype;
    }

    public ClassifierTrainingSet withPhenotype(String phenotype) {
        this.phenotype = phenotype;
        return this;
    }

    @JsonProperty("genome_name")
    public String getGenomeName() {
        return genomeName;
    }

    @JsonProperty("genome_name")
    public void setGenomeName(String genomeName) {
        this.genomeName = genomeName;
    }

    public ClassifierTrainingSet withGenomeName(String genomeName) {
        this.genomeName = genomeName;
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
        return ((((((("ClassifierTrainingSet"+" [phenotype=")+ phenotype)+", genomeName=")+ genomeName)+", additionalProperties=")+ additionalProperties)+"]");
    }

}
