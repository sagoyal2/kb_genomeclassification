
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
 * <p>Original spec-file type: KNearestNeighborsOptions</p>
 * 
 * 
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@Generated("com.googlecode.jsonschema2pojo")
@JsonPropertyOrder({
    "n_neighbors",
    "weights",
    "algorithm",
    "leaf_size",
    "p",
    "metric",
    "metric_params",
    "knn_n_jobs"
})
public class KNearestNeighborsOptions {

    @JsonProperty("n_neighbors")
    private Long nNeighbors;
    @JsonProperty("weights")
    private String weights;
    @JsonProperty("algorithm")
    private String algorithm;
    @JsonProperty("leaf_size")
    private Long leafSize;
    @JsonProperty("p")
    private Long p;
    @JsonProperty("metric")
    private String metric;
    @JsonProperty("metric_params")
    private String metricParams;
    @JsonProperty("knn_n_jobs")
    private Long knnNJobs;
    private Map<String, Object> additionalProperties = new HashMap<String, Object>();

    @JsonProperty("n_neighbors")
    public Long getNNeighbors() {
        return nNeighbors;
    }

    @JsonProperty("n_neighbors")
    public void setNNeighbors(Long nNeighbors) {
        this.nNeighbors = nNeighbors;
    }

    public KNearestNeighborsOptions withNNeighbors(Long nNeighbors) {
        this.nNeighbors = nNeighbors;
        return this;
    }

    @JsonProperty("weights")
    public String getWeights() {
        return weights;
    }

    @JsonProperty("weights")
    public void setWeights(String weights) {
        this.weights = weights;
    }

    public KNearestNeighborsOptions withWeights(String weights) {
        this.weights = weights;
        return this;
    }

    @JsonProperty("algorithm")
    public String getAlgorithm() {
        return algorithm;
    }

    @JsonProperty("algorithm")
    public void setAlgorithm(String algorithm) {
        this.algorithm = algorithm;
    }

    public KNearestNeighborsOptions withAlgorithm(String algorithm) {
        this.algorithm = algorithm;
        return this;
    }

    @JsonProperty("leaf_size")
    public Long getLeafSize() {
        return leafSize;
    }

    @JsonProperty("leaf_size")
    public void setLeafSize(Long leafSize) {
        this.leafSize = leafSize;
    }

    public KNearestNeighborsOptions withLeafSize(Long leafSize) {
        this.leafSize = leafSize;
        return this;
    }

    @JsonProperty("p")
    public Long getP() {
        return p;
    }

    @JsonProperty("p")
    public void setP(Long p) {
        this.p = p;
    }

    public KNearestNeighborsOptions withP(Long p) {
        this.p = p;
        return this;
    }

    @JsonProperty("metric")
    public String getMetric() {
        return metric;
    }

    @JsonProperty("metric")
    public void setMetric(String metric) {
        this.metric = metric;
    }

    public KNearestNeighborsOptions withMetric(String metric) {
        this.metric = metric;
        return this;
    }

    @JsonProperty("metric_params")
    public String getMetricParams() {
        return metricParams;
    }

    @JsonProperty("metric_params")
    public void setMetricParams(String metricParams) {
        this.metricParams = metricParams;
    }

    public KNearestNeighborsOptions withMetricParams(String metricParams) {
        this.metricParams = metricParams;
        return this;
    }

    @JsonProperty("knn_n_jobs")
    public Long getKnnNJobs() {
        return knnNJobs;
    }

    @JsonProperty("knn_n_jobs")
    public void setKnnNJobs(Long knnNJobs) {
        this.knnNJobs = knnNJobs;
    }

    public KNearestNeighborsOptions withKnnNJobs(Long knnNJobs) {
        this.knnNJobs = knnNJobs;
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
        return ((((((((((((((((((("KNearestNeighborsOptions"+" [nNeighbors=")+ nNeighbors)+", weights=")+ weights)+", algorithm=")+ algorithm)+", leafSize=")+ leafSize)+", p=")+ p)+", metric=")+ metric)+", metricParams=")+ metricParams)+", knnNJobs=")+ knnNJobs)+", additionalProperties=")+ additionalProperties)+"]");
    }

}
