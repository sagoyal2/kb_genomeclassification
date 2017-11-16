
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
 * <p>Original spec-file type: ClassifierPredictionOutput</p>
 * 
 * 
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@Generated("com.googlecode.jsonschema2pojo")
@JsonPropertyOrder({
    "prediction_accuracy",
    "predictions"
})
public class ClassifierPredictionOutput {

    @JsonProperty("prediction_accuracy")
    private Double predictionAccuracy;
    @JsonProperty("predictions")
    private Map<String, String> predictions;
    private Map<java.lang.String, Object> additionalProperties = new HashMap<java.lang.String, Object>();

    @JsonProperty("prediction_accuracy")
    public Double getPredictionAccuracy() {
        return predictionAccuracy;
    }

    @JsonProperty("prediction_accuracy")
    public void setPredictionAccuracy(Double predictionAccuracy) {
        this.predictionAccuracy = predictionAccuracy;
    }

    public ClassifierPredictionOutput withPredictionAccuracy(Double predictionAccuracy) {
        this.predictionAccuracy = predictionAccuracy;
        return this;
    }

    @JsonProperty("predictions")
    public Map<String, String> getPredictions() {
        return predictions;
    }

    @JsonProperty("predictions")
    public void setPredictions(Map<String, String> predictions) {
        this.predictions = predictions;
    }

    public ClassifierPredictionOutput withPredictions(Map<String, String> predictions) {
        this.predictions = predictions;
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
        return ((((((("ClassifierPredictionOutput"+" [predictionAccuracy=")+ predictionAccuracy)+", predictions=")+ predictions)+", additionalProperties=")+ additionalProperties)+"]");
    }

}
