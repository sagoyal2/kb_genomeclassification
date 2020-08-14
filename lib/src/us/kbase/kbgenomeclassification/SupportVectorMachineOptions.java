
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
 * <p>Original spec-file type: SupportVectorMachineOptions</p>
 * 
 * 
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@Generated("com.googlecode.jsonschema2pojo")
@JsonPropertyOrder({
    "svm_C",
    "kernel",
    "degree",
    "gamma",
    "coef0",
    "probability",
    "shrinking",
    "svm_tolerance",
    "cache_size",
    "svm_class_weight",
    "svm_verbose",
    "svm_max_iter",
    "decision_function_shape",
    "svm_random_state"
})
public class SupportVectorMachineOptions {

    @JsonProperty("svm_C")
    private Double svmC;
    @JsonProperty("kernel")
    private String kernel;
    @JsonProperty("degree")
    private Long degree;
    @JsonProperty("gamma")
    private String gamma;
    @JsonProperty("coef0")
    private Double coef0;
    @JsonProperty("probability")
    private String probability;
    @JsonProperty("shrinking")
    private String shrinking;
    @JsonProperty("svm_tolerance")
    private Double svmTolerance;
    @JsonProperty("cache_size")
    private Double cacheSize;
    @JsonProperty("svm_class_weight")
    private String svmClassWeight;
    @JsonProperty("svm_verbose")
    private String svmVerbose;
    @JsonProperty("svm_max_iter")
    private Long svmMaxIter;
    @JsonProperty("decision_function_shape")
    private String decisionFunctionShape;
    @JsonProperty("svm_random_state")
    private Long svmRandomState;
    private Map<String, Object> additionalProperties = new HashMap<String, Object>();

    @JsonProperty("svm_C")
    public Double getSvmC() {
        return svmC;
    }

    @JsonProperty("svm_C")
    public void setSvmC(Double svmC) {
        this.svmC = svmC;
    }

    public SupportVectorMachineOptions withSvmC(Double svmC) {
        this.svmC = svmC;
        return this;
    }

    @JsonProperty("kernel")
    public String getKernel() {
        return kernel;
    }

    @JsonProperty("kernel")
    public void setKernel(String kernel) {
        this.kernel = kernel;
    }

    public SupportVectorMachineOptions withKernel(String kernel) {
        this.kernel = kernel;
        return this;
    }

    @JsonProperty("degree")
    public Long getDegree() {
        return degree;
    }

    @JsonProperty("degree")
    public void setDegree(Long degree) {
        this.degree = degree;
    }

    public SupportVectorMachineOptions withDegree(Long degree) {
        this.degree = degree;
        return this;
    }

    @JsonProperty("gamma")
    public String getGamma() {
        return gamma;
    }

    @JsonProperty("gamma")
    public void setGamma(String gamma) {
        this.gamma = gamma;
    }

    public SupportVectorMachineOptions withGamma(String gamma) {
        this.gamma = gamma;
        return this;
    }

    @JsonProperty("coef0")
    public Double getCoef0() {
        return coef0;
    }

    @JsonProperty("coef0")
    public void setCoef0(Double coef0) {
        this.coef0 = coef0;
    }

    public SupportVectorMachineOptions withCoef0(Double coef0) {
        this.coef0 = coef0;
        return this;
    }

    @JsonProperty("probability")
    public String getProbability() {
        return probability;
    }

    @JsonProperty("probability")
    public void setProbability(String probability) {
        this.probability = probability;
    }

    public SupportVectorMachineOptions withProbability(String probability) {
        this.probability = probability;
        return this;
    }

    @JsonProperty("shrinking")
    public String getShrinking() {
        return shrinking;
    }

    @JsonProperty("shrinking")
    public void setShrinking(String shrinking) {
        this.shrinking = shrinking;
    }

    public SupportVectorMachineOptions withShrinking(String shrinking) {
        this.shrinking = shrinking;
        return this;
    }

    @JsonProperty("svm_tolerance")
    public Double getSvmTolerance() {
        return svmTolerance;
    }

    @JsonProperty("svm_tolerance")
    public void setSvmTolerance(Double svmTolerance) {
        this.svmTolerance = svmTolerance;
    }

    public SupportVectorMachineOptions withSvmTolerance(Double svmTolerance) {
        this.svmTolerance = svmTolerance;
        return this;
    }

    @JsonProperty("cache_size")
    public Double getCacheSize() {
        return cacheSize;
    }

    @JsonProperty("cache_size")
    public void setCacheSize(Double cacheSize) {
        this.cacheSize = cacheSize;
    }

    public SupportVectorMachineOptions withCacheSize(Double cacheSize) {
        this.cacheSize = cacheSize;
        return this;
    }

    @JsonProperty("svm_class_weight")
    public String getSvmClassWeight() {
        return svmClassWeight;
    }

    @JsonProperty("svm_class_weight")
    public void setSvmClassWeight(String svmClassWeight) {
        this.svmClassWeight = svmClassWeight;
    }

    public SupportVectorMachineOptions withSvmClassWeight(String svmClassWeight) {
        this.svmClassWeight = svmClassWeight;
        return this;
    }

    @JsonProperty("svm_verbose")
    public String getSvmVerbose() {
        return svmVerbose;
    }

    @JsonProperty("svm_verbose")
    public void setSvmVerbose(String svmVerbose) {
        this.svmVerbose = svmVerbose;
    }

    public SupportVectorMachineOptions withSvmVerbose(String svmVerbose) {
        this.svmVerbose = svmVerbose;
        return this;
    }

    @JsonProperty("svm_max_iter")
    public Long getSvmMaxIter() {
        return svmMaxIter;
    }

    @JsonProperty("svm_max_iter")
    public void setSvmMaxIter(Long svmMaxIter) {
        this.svmMaxIter = svmMaxIter;
    }

    public SupportVectorMachineOptions withSvmMaxIter(Long svmMaxIter) {
        this.svmMaxIter = svmMaxIter;
        return this;
    }

    @JsonProperty("decision_function_shape")
    public String getDecisionFunctionShape() {
        return decisionFunctionShape;
    }

    @JsonProperty("decision_function_shape")
    public void setDecisionFunctionShape(String decisionFunctionShape) {
        this.decisionFunctionShape = decisionFunctionShape;
    }

    public SupportVectorMachineOptions withDecisionFunctionShape(String decisionFunctionShape) {
        this.decisionFunctionShape = decisionFunctionShape;
        return this;
    }

    @JsonProperty("svm_random_state")
    public Long getSvmRandomState() {
        return svmRandomState;
    }

    @JsonProperty("svm_random_state")
    public void setSvmRandomState(Long svmRandomState) {
        this.svmRandomState = svmRandomState;
    }

    public SupportVectorMachineOptions withSvmRandomState(Long svmRandomState) {
        this.svmRandomState = svmRandomState;
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
        return ((((((((((((((((((((((((((((((("SupportVectorMachineOptions"+" [svmC=")+ svmC)+", kernel=")+ kernel)+", degree=")+ degree)+", gamma=")+ gamma)+", coef0=")+ coef0)+", probability=")+ probability)+", shrinking=")+ shrinking)+", svmTolerance=")+ svmTolerance)+", cacheSize=")+ cacheSize)+", svmClassWeight=")+ svmClassWeight)+", svmVerbose=")+ svmVerbose)+", svmMaxIter=")+ svmMaxIter)+", decisionFunctionShape=")+ decisionFunctionShape)+", svmRandomState=")+ svmRandomState)+", additionalProperties=")+ additionalProperties)+"]");
    }

}
