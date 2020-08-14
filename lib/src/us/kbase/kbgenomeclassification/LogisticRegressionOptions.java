
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
 * <p>Original spec-file type: LogisticRegressionOptions</p>
 * 
 * 
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@Generated("com.googlecode.jsonschema2pojo")
@JsonPropertyOrder({
    "penalty",
    "dual",
    "lr_tolerance",
    "lr_C",
    "fit_intercept",
    "intercept_scaling",
    "lr_class_weight",
    "lr_random_state",
    "lr_solver",
    "lr_max_iter",
    "multi_class",
    "lr_verbose",
    "lr_warm_start",
    "lr_n_jobs"
})
public class LogisticRegressionOptions {

    @JsonProperty("penalty")
    private String penalty;
    @JsonProperty("dual")
    private String dual;
    @JsonProperty("lr_tolerance")
    private Double lrTolerance;
    @JsonProperty("lr_C")
    private Double lrC;
    @JsonProperty("fit_intercept")
    private String fitIntercept;
    @JsonProperty("intercept_scaling")
    private Double interceptScaling;
    @JsonProperty("lr_class_weight")
    private String lrClassWeight;
    @JsonProperty("lr_random_state")
    private Long lrRandomState;
    @JsonProperty("lr_solver")
    private String lrSolver;
    @JsonProperty("lr_max_iter")
    private Long lrMaxIter;
    @JsonProperty("multi_class")
    private String multiClass;
    @JsonProperty("lr_verbose")
    private String lrVerbose;
    @JsonProperty("lr_warm_start")
    private Long lrWarmStart;
    @JsonProperty("lr_n_jobs")
    private Long lrNJobs;
    private Map<String, Object> additionalProperties = new HashMap<String, Object>();

    @JsonProperty("penalty")
    public String getPenalty() {
        return penalty;
    }

    @JsonProperty("penalty")
    public void setPenalty(String penalty) {
        this.penalty = penalty;
    }

    public LogisticRegressionOptions withPenalty(String penalty) {
        this.penalty = penalty;
        return this;
    }

    @JsonProperty("dual")
    public String getDual() {
        return dual;
    }

    @JsonProperty("dual")
    public void setDual(String dual) {
        this.dual = dual;
    }

    public LogisticRegressionOptions withDual(String dual) {
        this.dual = dual;
        return this;
    }

    @JsonProperty("lr_tolerance")
    public Double getLrTolerance() {
        return lrTolerance;
    }

    @JsonProperty("lr_tolerance")
    public void setLrTolerance(Double lrTolerance) {
        this.lrTolerance = lrTolerance;
    }

    public LogisticRegressionOptions withLrTolerance(Double lrTolerance) {
        this.lrTolerance = lrTolerance;
        return this;
    }

    @JsonProperty("lr_C")
    public Double getLrC() {
        return lrC;
    }

    @JsonProperty("lr_C")
    public void setLrC(Double lrC) {
        this.lrC = lrC;
    }

    public LogisticRegressionOptions withLrC(Double lrC) {
        this.lrC = lrC;
        return this;
    }

    @JsonProperty("fit_intercept")
    public String getFitIntercept() {
        return fitIntercept;
    }

    @JsonProperty("fit_intercept")
    public void setFitIntercept(String fitIntercept) {
        this.fitIntercept = fitIntercept;
    }

    public LogisticRegressionOptions withFitIntercept(String fitIntercept) {
        this.fitIntercept = fitIntercept;
        return this;
    }

    @JsonProperty("intercept_scaling")
    public Double getInterceptScaling() {
        return interceptScaling;
    }

    @JsonProperty("intercept_scaling")
    public void setInterceptScaling(Double interceptScaling) {
        this.interceptScaling = interceptScaling;
    }

    public LogisticRegressionOptions withInterceptScaling(Double interceptScaling) {
        this.interceptScaling = interceptScaling;
        return this;
    }

    @JsonProperty("lr_class_weight")
    public String getLrClassWeight() {
        return lrClassWeight;
    }

    @JsonProperty("lr_class_weight")
    public void setLrClassWeight(String lrClassWeight) {
        this.lrClassWeight = lrClassWeight;
    }

    public LogisticRegressionOptions withLrClassWeight(String lrClassWeight) {
        this.lrClassWeight = lrClassWeight;
        return this;
    }

    @JsonProperty("lr_random_state")
    public Long getLrRandomState() {
        return lrRandomState;
    }

    @JsonProperty("lr_random_state")
    public void setLrRandomState(Long lrRandomState) {
        this.lrRandomState = lrRandomState;
    }

    public LogisticRegressionOptions withLrRandomState(Long lrRandomState) {
        this.lrRandomState = lrRandomState;
        return this;
    }

    @JsonProperty("lr_solver")
    public String getLrSolver() {
        return lrSolver;
    }

    @JsonProperty("lr_solver")
    public void setLrSolver(String lrSolver) {
        this.lrSolver = lrSolver;
    }

    public LogisticRegressionOptions withLrSolver(String lrSolver) {
        this.lrSolver = lrSolver;
        return this;
    }

    @JsonProperty("lr_max_iter")
    public Long getLrMaxIter() {
        return lrMaxIter;
    }

    @JsonProperty("lr_max_iter")
    public void setLrMaxIter(Long lrMaxIter) {
        this.lrMaxIter = lrMaxIter;
    }

    public LogisticRegressionOptions withLrMaxIter(Long lrMaxIter) {
        this.lrMaxIter = lrMaxIter;
        return this;
    }

    @JsonProperty("multi_class")
    public String getMultiClass() {
        return multiClass;
    }

    @JsonProperty("multi_class")
    public void setMultiClass(String multiClass) {
        this.multiClass = multiClass;
    }

    public LogisticRegressionOptions withMultiClass(String multiClass) {
        this.multiClass = multiClass;
        return this;
    }

    @JsonProperty("lr_verbose")
    public String getLrVerbose() {
        return lrVerbose;
    }

    @JsonProperty("lr_verbose")
    public void setLrVerbose(String lrVerbose) {
        this.lrVerbose = lrVerbose;
    }

    public LogisticRegressionOptions withLrVerbose(String lrVerbose) {
        this.lrVerbose = lrVerbose;
        return this;
    }

    @JsonProperty("lr_warm_start")
    public Long getLrWarmStart() {
        return lrWarmStart;
    }

    @JsonProperty("lr_warm_start")
    public void setLrWarmStart(Long lrWarmStart) {
        this.lrWarmStart = lrWarmStart;
    }

    public LogisticRegressionOptions withLrWarmStart(Long lrWarmStart) {
        this.lrWarmStart = lrWarmStart;
        return this;
    }

    @JsonProperty("lr_n_jobs")
    public Long getLrNJobs() {
        return lrNJobs;
    }

    @JsonProperty("lr_n_jobs")
    public void setLrNJobs(Long lrNJobs) {
        this.lrNJobs = lrNJobs;
    }

    public LogisticRegressionOptions withLrNJobs(Long lrNJobs) {
        this.lrNJobs = lrNJobs;
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
        return ((((((((((((((((((((((((((((((("LogisticRegressionOptions"+" [penalty=")+ penalty)+", dual=")+ dual)+", lrTolerance=")+ lrTolerance)+", lrC=")+ lrC)+", fitIntercept=")+ fitIntercept)+", interceptScaling=")+ interceptScaling)+", lrClassWeight=")+ lrClassWeight)+", lrRandomState=")+ lrRandomState)+", lrSolver=")+ lrSolver)+", lrMaxIter=")+ lrMaxIter)+", multiClass=")+ multiClass)+", lrVerbose=")+ lrVerbose)+", lrWarmStart=")+ lrWarmStart)+", lrNJobs=")+ lrNJobs)+", additionalProperties=")+ additionalProperties)+"]");
    }

}
