
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
 * <p>Original spec-file type: PredictedPhenotypeOut</p>
 * 
 * 
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@Generated("com.googlecode.jsonschema2pojo")
@JsonPropertyOrder({
    "prediction_probabilities",
    "phenotype",
    "genome_name",
    "genome_ref"
})
public class PredictedPhenotypeOut {

    @JsonProperty("prediction_probabilities")
    private Double predictionProbabilities;
    @JsonProperty("phenotype")
    private String phenotype;
    @JsonProperty("genome_name")
    private String genomeName;
    @JsonProperty("genome_ref")
    private String genomeRef;
    private Map<String, Object> additionalProperties = new HashMap<String, Object>();

    @JsonProperty("prediction_probabilities")
    public Double getPredictionProbabilities() {
        return predictionProbabilities;
    }

    @JsonProperty("prediction_probabilities")
    public void setPredictionProbabilities(Double predictionProbabilities) {
        this.predictionProbabilities = predictionProbabilities;
    }

    public PredictedPhenotypeOut withPredictionProbabilities(Double predictionProbabilities) {
        this.predictionProbabilities = predictionProbabilities;
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

    public PredictedPhenotypeOut withPhenotype(String phenotype) {
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

    public PredictedPhenotypeOut withGenomeName(String genomeName) {
        this.genomeName = genomeName;
        return this;
    }

    @JsonProperty("genome_ref")
    public String getGenomeRef() {
        return genomeRef;
    }

    @JsonProperty("genome_ref")
    public void setGenomeRef(String genomeRef) {
        this.genomeRef = genomeRef;
    }

    public PredictedPhenotypeOut withGenomeRef(String genomeRef) {
        this.genomeRef = genomeRef;
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
        return ((((((((((("PredictedPhenotypeOut"+" [predictionProbabilities=")+ predictionProbabilities)+", phenotype=")+ phenotype)+", genomeName=")+ genomeName)+", genomeRef=")+ genomeRef)+", additionalProperties=")+ additionalProperties)+"]");
    }

}
