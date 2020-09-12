
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
 * <p>Original spec-file type: GaussianNBOptions</p>
 * 
 * 
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@Generated("com.googlecode.jsonschema2pojo")
@JsonPropertyOrder({
    "priors"
})
public class GaussianNBOptions {

    @JsonProperty("priors")
    private String priors;
    private Map<String, Object> additionalProperties = new HashMap<String, Object>();

    @JsonProperty("priors")
    public String getPriors() {
        return priors;
    }

    @JsonProperty("priors")
    public void setPriors(String priors) {
        this.priors = priors;
    }

    public GaussianNBOptions withPriors(String priors) {
        this.priors = priors;
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
        return ((((("GaussianNBOptions"+" [priors=")+ priors)+", additionalProperties=")+ additionalProperties)+"]");
    }

}
