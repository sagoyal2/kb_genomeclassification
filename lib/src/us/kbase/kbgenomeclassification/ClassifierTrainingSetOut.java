
package us.kbase.kbgenomeclassification;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Generated;
import com.fasterxml.jackson.annotation.JsonAnyGetter;
import com.fasterxml.jackson.annotation.JsonAnySetter;
import com.fasterxml.jackson.annotation.JsonInclude;
import com.fasterxml.jackson.annotation.JsonProperty;
import com.fasterxml.jackson.annotation.JsonPropertyOrder;


/**
 * <p>Original spec-file type: ClassifierTrainingSetOut</p>
 * 
 * 
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@Generated("com.googlecode.jsonschema2pojo")
@JsonPropertyOrder({
    "phenotype",
    "genome_name",
    "genome_ref",
    "references",
    "evidence_types"
})
public class ClassifierTrainingSetOut {

    @JsonProperty("phenotype")
    private java.lang.String phenotype;
    @JsonProperty("genome_name")
    private java.lang.String genomeName;
    @JsonProperty("genome_ref")
    private java.lang.String genomeRef;
    @JsonProperty("references")
    private List<String> references;
    @JsonProperty("evidence_types")
    private List<String> evidenceTypes;
    private Map<java.lang.String, Object> additionalProperties = new HashMap<java.lang.String, Object>();

    @JsonProperty("phenotype")
    public java.lang.String getPhenotype() {
        return phenotype;
    }

    @JsonProperty("phenotype")
    public void setPhenotype(java.lang.String phenotype) {
        this.phenotype = phenotype;
    }

    public ClassifierTrainingSetOut withPhenotype(java.lang.String phenotype) {
        this.phenotype = phenotype;
        return this;
    }

    @JsonProperty("genome_name")
    public java.lang.String getGenomeName() {
        return genomeName;
    }

    @JsonProperty("genome_name")
    public void setGenomeName(java.lang.String genomeName) {
        this.genomeName = genomeName;
    }

    public ClassifierTrainingSetOut withGenomeName(java.lang.String genomeName) {
        this.genomeName = genomeName;
        return this;
    }

    @JsonProperty("genome_ref")
    public java.lang.String getGenomeRef() {
        return genomeRef;
    }

    @JsonProperty("genome_ref")
    public void setGenomeRef(java.lang.String genomeRef) {
        this.genomeRef = genomeRef;
    }

    public ClassifierTrainingSetOut withGenomeRef(java.lang.String genomeRef) {
        this.genomeRef = genomeRef;
        return this;
    }

    @JsonProperty("references")
    public List<String> getReferences() {
        return references;
    }

    @JsonProperty("references")
    public void setReferences(List<String> references) {
        this.references = references;
    }

    public ClassifierTrainingSetOut withReferences(List<String> references) {
        this.references = references;
        return this;
    }

    @JsonProperty("evidence_types")
    public List<String> getEvidenceTypes() {
        return evidenceTypes;
    }

    @JsonProperty("evidence_types")
    public void setEvidenceTypes(List<String> evidenceTypes) {
        this.evidenceTypes = evidenceTypes;
    }

    public ClassifierTrainingSetOut withEvidenceTypes(List<String> evidenceTypes) {
        this.evidenceTypes = evidenceTypes;
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
        return ((((((((((((("ClassifierTrainingSetOut"+" [phenotype=")+ phenotype)+", genomeName=")+ genomeName)+", genomeRef=")+ genomeRef)+", references=")+ references)+", evidenceTypes=")+ evidenceTypes)+", additionalProperties=")+ additionalProperties)+"]");
    }

}
