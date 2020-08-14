
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
 * <p>Original spec-file type: NeuralNetworkOptions</p>
 * 
 * 
 */
@JsonInclude(JsonInclude.Include.NON_NULL)
@Generated("com.googlecode.jsonschema2pojo")
@JsonPropertyOrder({
    "hidden_layer_sizes",
    "activation",
    "mlp_solver",
    "alpha",
    "batch_size",
    "learning_rate",
    "learning_rate_init",
    "power_t",
    "mlp_max_iter",
    "shuffle",
    "mlp_random_state",
    "mlp_tolerance",
    "mlp_verbose",
    "mlp_warm_start",
    "momentum",
    "nesterovs_momentum",
    "early_stopping",
    "validation_fraction",
    "beta_1",
    "beta_2",
    "epsilon"
})
public class NeuralNetworkOptions {

    @JsonProperty("hidden_layer_sizes")
    private String hiddenLayerSizes;
    @JsonProperty("activation")
    private String activation;
    @JsonProperty("mlp_solver")
    private String mlpSolver;
    @JsonProperty("alpha")
    private Double alpha;
    @JsonProperty("batch_size")
    private String batchSize;
    @JsonProperty("learning_rate")
    private String learningRate;
    @JsonProperty("learning_rate_init")
    private Double learningRateInit;
    @JsonProperty("power_t")
    private Double powerT;
    @JsonProperty("mlp_max_iter")
    private Long mlpMaxIter;
    @JsonProperty("shuffle")
    private String shuffle;
    @JsonProperty("mlp_random_state")
    private Long mlpRandomState;
    @JsonProperty("mlp_tolerance")
    private Double mlpTolerance;
    @JsonProperty("mlp_verbose")
    private String mlpVerbose;
    @JsonProperty("mlp_warm_start")
    private String mlpWarmStart;
    @JsonProperty("momentum")
    private Double momentum;
    @JsonProperty("nesterovs_momentum")
    private String nesterovsMomentum;
    @JsonProperty("early_stopping")
    private String earlyStopping;
    @JsonProperty("validation_fraction")
    private Double validationFraction;
    @JsonProperty("beta_1")
    private Double beta1;
    @JsonProperty("beta_2")
    private Double beta2;
    @JsonProperty("epsilon")
    private Double epsilon;
    private Map<String, Object> additionalProperties = new HashMap<String, Object>();

    @JsonProperty("hidden_layer_sizes")
    public String getHiddenLayerSizes() {
        return hiddenLayerSizes;
    }

    @JsonProperty("hidden_layer_sizes")
    public void setHiddenLayerSizes(String hiddenLayerSizes) {
        this.hiddenLayerSizes = hiddenLayerSizes;
    }

    public NeuralNetworkOptions withHiddenLayerSizes(String hiddenLayerSizes) {
        this.hiddenLayerSizes = hiddenLayerSizes;
        return this;
    }

    @JsonProperty("activation")
    public String getActivation() {
        return activation;
    }

    @JsonProperty("activation")
    public void setActivation(String activation) {
        this.activation = activation;
    }

    public NeuralNetworkOptions withActivation(String activation) {
        this.activation = activation;
        return this;
    }

    @JsonProperty("mlp_solver")
    public String getMlpSolver() {
        return mlpSolver;
    }

    @JsonProperty("mlp_solver")
    public void setMlpSolver(String mlpSolver) {
        this.mlpSolver = mlpSolver;
    }

    public NeuralNetworkOptions withMlpSolver(String mlpSolver) {
        this.mlpSolver = mlpSolver;
        return this;
    }

    @JsonProperty("alpha")
    public Double getAlpha() {
        return alpha;
    }

    @JsonProperty("alpha")
    public void setAlpha(Double alpha) {
        this.alpha = alpha;
    }

    public NeuralNetworkOptions withAlpha(Double alpha) {
        this.alpha = alpha;
        return this;
    }

    @JsonProperty("batch_size")
    public String getBatchSize() {
        return batchSize;
    }

    @JsonProperty("batch_size")
    public void setBatchSize(String batchSize) {
        this.batchSize = batchSize;
    }

    public NeuralNetworkOptions withBatchSize(String batchSize) {
        this.batchSize = batchSize;
        return this;
    }

    @JsonProperty("learning_rate")
    public String getLearningRate() {
        return learningRate;
    }

    @JsonProperty("learning_rate")
    public void setLearningRate(String learningRate) {
        this.learningRate = learningRate;
    }

    public NeuralNetworkOptions withLearningRate(String learningRate) {
        this.learningRate = learningRate;
        return this;
    }

    @JsonProperty("learning_rate_init")
    public Double getLearningRateInit() {
        return learningRateInit;
    }

    @JsonProperty("learning_rate_init")
    public void setLearningRateInit(Double learningRateInit) {
        this.learningRateInit = learningRateInit;
    }

    public NeuralNetworkOptions withLearningRateInit(Double learningRateInit) {
        this.learningRateInit = learningRateInit;
        return this;
    }

    @JsonProperty("power_t")
    public Double getPowerT() {
        return powerT;
    }

    @JsonProperty("power_t")
    public void setPowerT(Double powerT) {
        this.powerT = powerT;
    }

    public NeuralNetworkOptions withPowerT(Double powerT) {
        this.powerT = powerT;
        return this;
    }

    @JsonProperty("mlp_max_iter")
    public Long getMlpMaxIter() {
        return mlpMaxIter;
    }

    @JsonProperty("mlp_max_iter")
    public void setMlpMaxIter(Long mlpMaxIter) {
        this.mlpMaxIter = mlpMaxIter;
    }

    public NeuralNetworkOptions withMlpMaxIter(Long mlpMaxIter) {
        this.mlpMaxIter = mlpMaxIter;
        return this;
    }

    @JsonProperty("shuffle")
    public String getShuffle() {
        return shuffle;
    }

    @JsonProperty("shuffle")
    public void setShuffle(String shuffle) {
        this.shuffle = shuffle;
    }

    public NeuralNetworkOptions withShuffle(String shuffle) {
        this.shuffle = shuffle;
        return this;
    }

    @JsonProperty("mlp_random_state")
    public Long getMlpRandomState() {
        return mlpRandomState;
    }

    @JsonProperty("mlp_random_state")
    public void setMlpRandomState(Long mlpRandomState) {
        this.mlpRandomState = mlpRandomState;
    }

    public NeuralNetworkOptions withMlpRandomState(Long mlpRandomState) {
        this.mlpRandomState = mlpRandomState;
        return this;
    }

    @JsonProperty("mlp_tolerance")
    public Double getMlpTolerance() {
        return mlpTolerance;
    }

    @JsonProperty("mlp_tolerance")
    public void setMlpTolerance(Double mlpTolerance) {
        this.mlpTolerance = mlpTolerance;
    }

    public NeuralNetworkOptions withMlpTolerance(Double mlpTolerance) {
        this.mlpTolerance = mlpTolerance;
        return this;
    }

    @JsonProperty("mlp_verbose")
    public String getMlpVerbose() {
        return mlpVerbose;
    }

    @JsonProperty("mlp_verbose")
    public void setMlpVerbose(String mlpVerbose) {
        this.mlpVerbose = mlpVerbose;
    }

    public NeuralNetworkOptions withMlpVerbose(String mlpVerbose) {
        this.mlpVerbose = mlpVerbose;
        return this;
    }

    @JsonProperty("mlp_warm_start")
    public String getMlpWarmStart() {
        return mlpWarmStart;
    }

    @JsonProperty("mlp_warm_start")
    public void setMlpWarmStart(String mlpWarmStart) {
        this.mlpWarmStart = mlpWarmStart;
    }

    public NeuralNetworkOptions withMlpWarmStart(String mlpWarmStart) {
        this.mlpWarmStart = mlpWarmStart;
        return this;
    }

    @JsonProperty("momentum")
    public Double getMomentum() {
        return momentum;
    }

    @JsonProperty("momentum")
    public void setMomentum(Double momentum) {
        this.momentum = momentum;
    }

    public NeuralNetworkOptions withMomentum(Double momentum) {
        this.momentum = momentum;
        return this;
    }

    @JsonProperty("nesterovs_momentum")
    public String getNesterovsMomentum() {
        return nesterovsMomentum;
    }

    @JsonProperty("nesterovs_momentum")
    public void setNesterovsMomentum(String nesterovsMomentum) {
        this.nesterovsMomentum = nesterovsMomentum;
    }

    public NeuralNetworkOptions withNesterovsMomentum(String nesterovsMomentum) {
        this.nesterovsMomentum = nesterovsMomentum;
        return this;
    }

    @JsonProperty("early_stopping")
    public String getEarlyStopping() {
        return earlyStopping;
    }

    @JsonProperty("early_stopping")
    public void setEarlyStopping(String earlyStopping) {
        this.earlyStopping = earlyStopping;
    }

    public NeuralNetworkOptions withEarlyStopping(String earlyStopping) {
        this.earlyStopping = earlyStopping;
        return this;
    }

    @JsonProperty("validation_fraction")
    public Double getValidationFraction() {
        return validationFraction;
    }

    @JsonProperty("validation_fraction")
    public void setValidationFraction(Double validationFraction) {
        this.validationFraction = validationFraction;
    }

    public NeuralNetworkOptions withValidationFraction(Double validationFraction) {
        this.validationFraction = validationFraction;
        return this;
    }

    @JsonProperty("beta_1")
    public Double getBeta1() {
        return beta1;
    }

    @JsonProperty("beta_1")
    public void setBeta1(Double beta1) {
        this.beta1 = beta1;
    }

    public NeuralNetworkOptions withBeta1(Double beta1) {
        this.beta1 = beta1;
        return this;
    }

    @JsonProperty("beta_2")
    public Double getBeta2() {
        return beta2;
    }

    @JsonProperty("beta_2")
    public void setBeta2(Double beta2) {
        this.beta2 = beta2;
    }

    public NeuralNetworkOptions withBeta2(Double beta2) {
        this.beta2 = beta2;
        return this;
    }

    @JsonProperty("epsilon")
    public Double getEpsilon() {
        return epsilon;
    }

    @JsonProperty("epsilon")
    public void setEpsilon(Double epsilon) {
        this.epsilon = epsilon;
    }

    public NeuralNetworkOptions withEpsilon(Double epsilon) {
        this.epsilon = epsilon;
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
        return ((((((((((((((((((((((((((((((((((((((((((((("NeuralNetworkOptions"+" [hiddenLayerSizes=")+ hiddenLayerSizes)+", activation=")+ activation)+", mlpSolver=")+ mlpSolver)+", alpha=")+ alpha)+", batchSize=")+ batchSize)+", learningRate=")+ learningRate)+", learningRateInit=")+ learningRateInit)+", powerT=")+ powerT)+", mlpMaxIter=")+ mlpMaxIter)+", shuffle=")+ shuffle)+", mlpRandomState=")+ mlpRandomState)+", mlpTolerance=")+ mlpTolerance)+", mlpVerbose=")+ mlpVerbose)+", mlpWarmStart=")+ mlpWarmStart)+", momentum=")+ momentum)+", nesterovsMomentum=")+ nesterovsMomentum)+", earlyStopping=")+ earlyStopping)+", validationFraction=")+ validationFraction)+", beta1=")+ beta1)+", beta2=")+ beta2)+", epsilon=")+ epsilon)+", additionalProperties=")+ additionalProperties)+"]");
    }

}
