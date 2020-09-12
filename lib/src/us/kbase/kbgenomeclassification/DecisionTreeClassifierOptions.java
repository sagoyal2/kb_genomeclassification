
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
 * <p>Original spec-file type: DecisionTreeClassifierOptions</p>
 * 
 * 
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@Generated("com.googlecode.jsonschema2pojo")
@JsonPropertyOrder({
    "criterion",
    "splitter",
    "max_depth",
    "min_samples_split",
    "min_samples_leaf",
    "min_weight_fraction_leaf",
    "max_features",
    "dt_random_state",
    "max_leaf_nodes",
    "min_impurity_decrease",
    "dt_class_weight",
    "presort"
})
public class DecisionTreeClassifierOptions {

    @JsonProperty("criterion")
    private String criterion;
    @JsonProperty("splitter")
    private String splitter;
    @JsonProperty("max_depth")
    private Long maxDepth;
    @JsonProperty("min_samples_split")
    private Long minSamplesSplit;
    @JsonProperty("min_samples_leaf")
    private Long minSamplesLeaf;
    @JsonProperty("min_weight_fraction_leaf")
    private Double minWeightFractionLeaf;
    @JsonProperty("max_features")
    private String maxFeatures;
    @JsonProperty("dt_random_state")
    private Long dtRandomState;
    @JsonProperty("max_leaf_nodes")
    private Long maxLeafNodes;
    @JsonProperty("min_impurity_decrease")
    private Double minImpurityDecrease;
    @JsonProperty("dt_class_weight")
    private String dtClassWeight;
    @JsonProperty("presort")
    private String presort;
    private Map<String, Object> additionalProperties = new HashMap<String, Object>();

    @JsonProperty("criterion")
    public String getCriterion() {
        return criterion;
    }

    @JsonProperty("criterion")
    public void setCriterion(String criterion) {
        this.criterion = criterion;
    }

    public DecisionTreeClassifierOptions withCriterion(String criterion) {
        this.criterion = criterion;
        return this;
    }

    @JsonProperty("splitter")
    public String getSplitter() {
        return splitter;
    }

    @JsonProperty("splitter")
    public void setSplitter(String splitter) {
        this.splitter = splitter;
    }

    public DecisionTreeClassifierOptions withSplitter(String splitter) {
        this.splitter = splitter;
        return this;
    }

    @JsonProperty("max_depth")
    public Long getMaxDepth() {
        return maxDepth;
    }

    @JsonProperty("max_depth")
    public void setMaxDepth(Long maxDepth) {
        this.maxDepth = maxDepth;
    }

    public DecisionTreeClassifierOptions withMaxDepth(Long maxDepth) {
        this.maxDepth = maxDepth;
        return this;
    }

    @JsonProperty("min_samples_split")
    public Long getMinSamplesSplit() {
        return minSamplesSplit;
    }

    @JsonProperty("min_samples_split")
    public void setMinSamplesSplit(Long minSamplesSplit) {
        this.minSamplesSplit = minSamplesSplit;
    }

    public DecisionTreeClassifierOptions withMinSamplesSplit(Long minSamplesSplit) {
        this.minSamplesSplit = minSamplesSplit;
        return this;
    }

    @JsonProperty("min_samples_leaf")
    public Long getMinSamplesLeaf() {
        return minSamplesLeaf;
    }

    @JsonProperty("min_samples_leaf")
    public void setMinSamplesLeaf(Long minSamplesLeaf) {
        this.minSamplesLeaf = minSamplesLeaf;
    }

    public DecisionTreeClassifierOptions withMinSamplesLeaf(Long minSamplesLeaf) {
        this.minSamplesLeaf = minSamplesLeaf;
        return this;
    }

    @JsonProperty("min_weight_fraction_leaf")
    public Double getMinWeightFractionLeaf() {
        return minWeightFractionLeaf;
    }

    @JsonProperty("min_weight_fraction_leaf")
    public void setMinWeightFractionLeaf(Double minWeightFractionLeaf) {
        this.minWeightFractionLeaf = minWeightFractionLeaf;
    }

    public DecisionTreeClassifierOptions withMinWeightFractionLeaf(Double minWeightFractionLeaf) {
        this.minWeightFractionLeaf = minWeightFractionLeaf;
        return this;
    }

    @JsonProperty("max_features")
    public String getMaxFeatures() {
        return maxFeatures;
    }

    @JsonProperty("max_features")
    public void setMaxFeatures(String maxFeatures) {
        this.maxFeatures = maxFeatures;
    }

    public DecisionTreeClassifierOptions withMaxFeatures(String maxFeatures) {
        this.maxFeatures = maxFeatures;
        return this;
    }

    @JsonProperty("dt_random_state")
    public Long getDtRandomState() {
        return dtRandomState;
    }

    @JsonProperty("dt_random_state")
    public void setDtRandomState(Long dtRandomState) {
        this.dtRandomState = dtRandomState;
    }

    public DecisionTreeClassifierOptions withDtRandomState(Long dtRandomState) {
        this.dtRandomState = dtRandomState;
        return this;
    }

    @JsonProperty("max_leaf_nodes")
    public Long getMaxLeafNodes() {
        return maxLeafNodes;
    }

    @JsonProperty("max_leaf_nodes")
    public void setMaxLeafNodes(Long maxLeafNodes) {
        this.maxLeafNodes = maxLeafNodes;
    }

    public DecisionTreeClassifierOptions withMaxLeafNodes(Long maxLeafNodes) {
        this.maxLeafNodes = maxLeafNodes;
        return this;
    }

    @JsonProperty("min_impurity_decrease")
    public Double getMinImpurityDecrease() {
        return minImpurityDecrease;
    }

    @JsonProperty("min_impurity_decrease")
    public void setMinImpurityDecrease(Double minImpurityDecrease) {
        this.minImpurityDecrease = minImpurityDecrease;
    }

    public DecisionTreeClassifierOptions withMinImpurityDecrease(Double minImpurityDecrease) {
        this.minImpurityDecrease = minImpurityDecrease;
        return this;
    }

    @JsonProperty("dt_class_weight")
    public String getDtClassWeight() {
        return dtClassWeight;
    }

    @JsonProperty("dt_class_weight")
    public void setDtClassWeight(String dtClassWeight) {
        this.dtClassWeight = dtClassWeight;
    }

    public DecisionTreeClassifierOptions withDtClassWeight(String dtClassWeight) {
        this.dtClassWeight = dtClassWeight;
        return this;
    }

    @JsonProperty("presort")
    public String getPresort() {
        return presort;
    }

    @JsonProperty("presort")
    public void setPresort(String presort) {
        this.presort = presort;
    }

    public DecisionTreeClassifierOptions withPresort(String presort) {
        this.presort = presort;
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
        return ((((((((((((((((((((((((((("DecisionTreeClassifierOptions"+" [criterion=")+ criterion)+", splitter=")+ splitter)+", maxDepth=")+ maxDepth)+", minSamplesSplit=")+ minSamplesSplit)+", minSamplesLeaf=")+ minSamplesLeaf)+", minWeightFractionLeaf=")+ minWeightFractionLeaf)+", maxFeatures=")+ maxFeatures)+", dtRandomState=")+ dtRandomState)+", maxLeafNodes=")+ maxLeafNodes)+", minImpurityDecrease=")+ minImpurityDecrease)+", dtClassWeight=")+ dtClassWeight)+", presort=")+ presort)+", additionalProperties=")+ additionalProperties)+"]");
    }

}
