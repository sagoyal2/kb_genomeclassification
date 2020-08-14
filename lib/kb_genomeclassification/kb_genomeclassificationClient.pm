package kb_genomeclassification::kb_genomeclassificationClient;

use JSON::RPC::Client;
use POSIX;
use strict;
use Data::Dumper;
use URI;
use Bio::KBase::Exceptions;
my $get_time = sub { time, 0 };
eval {
    require Time::HiRes;
    $get_time = sub { Time::HiRes::gettimeofday() };
};

use Bio::KBase::AuthToken;

# Client version should match Impl version
# This is a Semantic Version number,
# http://semver.org
our $VERSION = "0.1.0";

=head1 NAME

kb_genomeclassification::kb_genomeclassificationClient

=head1 DESCRIPTION


A KBase module: kb_genomeclassification
This module build a classifier and predict phenotypes based on the classifier Another line


=cut

sub new
{
    my($class, $url, @args) = @_;
    

    my $self = {
	client => kb_genomeclassification::kb_genomeclassificationClient::RpcClient->new,
	url => $url,
	headers => [],
    };

    chomp($self->{hostname} = `hostname`);
    $self->{hostname} ||= 'unknown-host';

    #
    # Set up for propagating KBRPC_TAG and KBRPC_METADATA environment variables through
    # to invoked services. If these values are not set, we create a new tag
    # and a metadata field with basic information about the invoking script.
    #
    if ($ENV{KBRPC_TAG})
    {
	$self->{kbrpc_tag} = $ENV{KBRPC_TAG};
    }
    else
    {
	my ($t, $us) = &$get_time();
	$us = sprintf("%06d", $us);
	my $ts = strftime("%Y-%m-%dT%H:%M:%S.${us}Z", gmtime $t);
	$self->{kbrpc_tag} = "C:$0:$self->{hostname}:$$:$ts";
    }
    push(@{$self->{headers}}, 'Kbrpc-Tag', $self->{kbrpc_tag});

    if ($ENV{KBRPC_METADATA})
    {
	$self->{kbrpc_metadata} = $ENV{KBRPC_METADATA};
	push(@{$self->{headers}}, 'Kbrpc-Metadata', $self->{kbrpc_metadata});
    }

    if ($ENV{KBRPC_ERROR_DEST})
    {
	$self->{kbrpc_error_dest} = $ENV{KBRPC_ERROR_DEST};
	push(@{$self->{headers}}, 'Kbrpc-Errordest', $self->{kbrpc_error_dest});
    }

    #
    # This module requires authentication.
    #
    # We create an auth token, passing through the arguments that we were (hopefully) given.

    {
	my %arg_hash2 = @args;
	if (exists $arg_hash2{"token"}) {
	    $self->{token} = $arg_hash2{"token"};
	} elsif (exists $arg_hash2{"user_id"}) {
	    my $token = Bio::KBase::AuthToken->new(@args);
	    if (!$token->error_message) {
	        $self->{token} = $token->token;
	    }
	}
	
	if (exists $self->{token})
	{
	    $self->{client}->{token} = $self->{token};
	}
    }

    my $ua = $self->{client}->ua;	 
    my $timeout = $ENV{CDMI_TIMEOUT} || (30 * 60);	 
    $ua->timeout($timeout);
    bless $self, $class;
    #    $self->_validate_version();
    return $self;
}




=head2 build_classifier

  $output = $obj->build_classifier($params)

=over 4

=item Parameter and return types

=begin html

<pre>
$params is a kb_genomeclassification.BuildClassifierInput
$output is a kb_genomeclassification.ClassifierOut
BuildClassifierInput is a reference to a hash where the following keys are defined:
	genome_attribute has a value which is a string
	workspace has a value which is a string
	training_set_name has a value which is a string
	classifier_training_set has a value which is a reference to a hash where the key is a string and the value is a kb_genomeclassification.ClassifierTrainingSet
	classifier_object_name has a value which is a string
	description has a value which is a string
	classifier_to_run has a value which is a string
	logistic_regression has a value which is a kb_genomeclassification.LogisticRegressionOptions
	decision_tree_classifier has a value which is a kb_genomeclassification.DecisionTreeClassifierOptions
	gaussian_nb has a value which is a kb_genomeclassification.GaussianNBOptions
	k_nearest_neighbors has a value which is a kb_genomeclassification.KNearestNeighborsOptions
	support_vector_machine has a value which is a kb_genomeclassification.SupportVectorMachineOptions
	neural_network has a value which is a kb_genomeclassification.NeuralNetworkOptions
	ensemble_model has a value which is a kb_genomeclassification.EnsembleModelOptions
ClassifierTrainingSet is a reference to a hash where the following keys are defined:
	phenotype has a value which is a string
	genome_name has a value which is a string
LogisticRegressionOptions is a reference to a hash where the following keys are defined:
	penalty has a value which is a string
	dual has a value which is a kb_genomeclassification.boolean
	lr_tolerance has a value which is a float
	lr_C has a value which is a float
	fit_intercept has a value which is a kb_genomeclassification.boolean
	intercept_scaling has a value which is a float
	lr_class_weight has a value which is a string
	lr_random_state has a value which is an int
	lr_solver has a value which is a string
	lr_max_iter has a value which is an int
	multi_class has a value which is a string
	lr_verbose has a value which is a kb_genomeclassification.boolean
	lr_warm_start has a value which is an int
	lr_n_jobs has a value which is an int
boolean is a string
DecisionTreeClassifierOptions is a reference to a hash where the following keys are defined:
	criterion has a value which is a string
	splitter has a value which is a string
	max_depth has a value which is an int
	min_samples_split has a value which is an int
	min_samples_leaf has a value which is an int
	min_weight_fraction_leaf has a value which is a float
	max_features has a value which is a string
	dt_random_state has a value which is an int
	max_leaf_nodes has a value which is an int
	min_impurity_decrease has a value which is a float
	dt_class_weight has a value which is a string
	presort has a value which is a string
GaussianNBOptions is a reference to a hash where the following keys are defined:
	priors has a value which is a string
KNearestNeighborsOptions is a reference to a hash where the following keys are defined:
	n_neighbors has a value which is an int
	weights has a value which is a string
	algorithm has a value which is a string
	leaf_size has a value which is an int
	p has a value which is an int
	metric has a value which is a string
	metric_params has a value which is a string
	knn_n_jobs has a value which is an int
SupportVectorMachineOptions is a reference to a hash where the following keys are defined:
	svm_C has a value which is a float
	kernel has a value which is a string
	degree has a value which is an int
	gamma has a value which is a string
	coef0 has a value which is a float
	probability has a value which is a kb_genomeclassification.boolean
	shrinking has a value which is a kb_genomeclassification.boolean
	svm_tolerance has a value which is a float
	cache_size has a value which is a float
	svm_class_weight has a value which is a string
	svm_verbose has a value which is a kb_genomeclassification.boolean
	svm_max_iter has a value which is an int
	decision_function_shape has a value which is a string
	svm_random_state has a value which is an int
NeuralNetworkOptions is a reference to a hash where the following keys are defined:
	hidden_layer_sizes has a value which is a string
	activation has a value which is a string
	mlp_solver has a value which is a string
	alpha has a value which is a float
	batch_size has a value which is a string
	learning_rate has a value which is a string
	learning_rate_init has a value which is a float
	power_t has a value which is a float
	mlp_max_iter has a value which is an int
	shuffle has a value which is a kb_genomeclassification.boolean
	mlp_random_state has a value which is an int
	mlp_tolerance has a value which is a float
	mlp_verbose has a value which is a kb_genomeclassification.boolean
	mlp_warm_start has a value which is a kb_genomeclassification.boolean
	momentum has a value which is a float
	nesterovs_momentum has a value which is a kb_genomeclassification.boolean
	early_stopping has a value which is a kb_genomeclassification.boolean
	validation_fraction has a value which is a float
	beta_1 has a value which is a float
	beta_2 has a value which is a float
	epsilon has a value which is a float
EnsembleModelOptions is a reference to a hash where the following keys are defined:
	k_nearest_neighbors_box has a value which is an int
	gaussian_nb_box has a value which is an int
	logistic_regression_box has a value which is an int
	decision_tree_classifier_box has a value which is an int
	support_vector_machine_box has a value which is an int
	neural_network_box has a value which is an int
	voting has a value which is a string
	en_weights has a value which is a string
	en_n_jobs has a value which is an int
	flatten_transform has a value which is a kb_genomeclassification.boolean
ClassifierOut is a reference to a hash where the following keys are defined:
	classifier_info has a value which is a reference to a list where each element is a kb_genomeclassification.classifierInfo
	report_name has a value which is a string
	report_ref has a value which is a string
classifierInfo is a reference to a hash where the following keys are defined:
	classifier_name has a value which is a string
	classifier_ref has a value which is a string
	accuracy has a value which is a float

</pre>

=end html

=begin text

$params is a kb_genomeclassification.BuildClassifierInput
$output is a kb_genomeclassification.ClassifierOut
BuildClassifierInput is a reference to a hash where the following keys are defined:
	genome_attribute has a value which is a string
	workspace has a value which is a string
	training_set_name has a value which is a string
	classifier_training_set has a value which is a reference to a hash where the key is a string and the value is a kb_genomeclassification.ClassifierTrainingSet
	classifier_object_name has a value which is a string
	description has a value which is a string
	classifier_to_run has a value which is a string
	logistic_regression has a value which is a kb_genomeclassification.LogisticRegressionOptions
	decision_tree_classifier has a value which is a kb_genomeclassification.DecisionTreeClassifierOptions
	gaussian_nb has a value which is a kb_genomeclassification.GaussianNBOptions
	k_nearest_neighbors has a value which is a kb_genomeclassification.KNearestNeighborsOptions
	support_vector_machine has a value which is a kb_genomeclassification.SupportVectorMachineOptions
	neural_network has a value which is a kb_genomeclassification.NeuralNetworkOptions
	ensemble_model has a value which is a kb_genomeclassification.EnsembleModelOptions
ClassifierTrainingSet is a reference to a hash where the following keys are defined:
	phenotype has a value which is a string
	genome_name has a value which is a string
LogisticRegressionOptions is a reference to a hash where the following keys are defined:
	penalty has a value which is a string
	dual has a value which is a kb_genomeclassification.boolean
	lr_tolerance has a value which is a float
	lr_C has a value which is a float
	fit_intercept has a value which is a kb_genomeclassification.boolean
	intercept_scaling has a value which is a float
	lr_class_weight has a value which is a string
	lr_random_state has a value which is an int
	lr_solver has a value which is a string
	lr_max_iter has a value which is an int
	multi_class has a value which is a string
	lr_verbose has a value which is a kb_genomeclassification.boolean
	lr_warm_start has a value which is an int
	lr_n_jobs has a value which is an int
boolean is a string
DecisionTreeClassifierOptions is a reference to a hash where the following keys are defined:
	criterion has a value which is a string
	splitter has a value which is a string
	max_depth has a value which is an int
	min_samples_split has a value which is an int
	min_samples_leaf has a value which is an int
	min_weight_fraction_leaf has a value which is a float
	max_features has a value which is a string
	dt_random_state has a value which is an int
	max_leaf_nodes has a value which is an int
	min_impurity_decrease has a value which is a float
	dt_class_weight has a value which is a string
	presort has a value which is a string
GaussianNBOptions is a reference to a hash where the following keys are defined:
	priors has a value which is a string
KNearestNeighborsOptions is a reference to a hash where the following keys are defined:
	n_neighbors has a value which is an int
	weights has a value which is a string
	algorithm has a value which is a string
	leaf_size has a value which is an int
	p has a value which is an int
	metric has a value which is a string
	metric_params has a value which is a string
	knn_n_jobs has a value which is an int
SupportVectorMachineOptions is a reference to a hash where the following keys are defined:
	svm_C has a value which is a float
	kernel has a value which is a string
	degree has a value which is an int
	gamma has a value which is a string
	coef0 has a value which is a float
	probability has a value which is a kb_genomeclassification.boolean
	shrinking has a value which is a kb_genomeclassification.boolean
	svm_tolerance has a value which is a float
	cache_size has a value which is a float
	svm_class_weight has a value which is a string
	svm_verbose has a value which is a kb_genomeclassification.boolean
	svm_max_iter has a value which is an int
	decision_function_shape has a value which is a string
	svm_random_state has a value which is an int
NeuralNetworkOptions is a reference to a hash where the following keys are defined:
	hidden_layer_sizes has a value which is a string
	activation has a value which is a string
	mlp_solver has a value which is a string
	alpha has a value which is a float
	batch_size has a value which is a string
	learning_rate has a value which is a string
	learning_rate_init has a value which is a float
	power_t has a value which is a float
	mlp_max_iter has a value which is an int
	shuffle has a value which is a kb_genomeclassification.boolean
	mlp_random_state has a value which is an int
	mlp_tolerance has a value which is a float
	mlp_verbose has a value which is a kb_genomeclassification.boolean
	mlp_warm_start has a value which is a kb_genomeclassification.boolean
	momentum has a value which is a float
	nesterovs_momentum has a value which is a kb_genomeclassification.boolean
	early_stopping has a value which is a kb_genomeclassification.boolean
	validation_fraction has a value which is a float
	beta_1 has a value which is a float
	beta_2 has a value which is a float
	epsilon has a value which is a float
EnsembleModelOptions is a reference to a hash where the following keys are defined:
	k_nearest_neighbors_box has a value which is an int
	gaussian_nb_box has a value which is an int
	logistic_regression_box has a value which is an int
	decision_tree_classifier_box has a value which is an int
	support_vector_machine_box has a value which is an int
	neural_network_box has a value which is an int
	voting has a value which is a string
	en_weights has a value which is a string
	en_n_jobs has a value which is an int
	flatten_transform has a value which is a kb_genomeclassification.boolean
ClassifierOut is a reference to a hash where the following keys are defined:
	classifier_info has a value which is a reference to a list where each element is a kb_genomeclassification.classifierInfo
	report_name has a value which is a string
	report_ref has a value which is a string
classifierInfo is a reference to a hash where the following keys are defined:
	classifier_name has a value which is a string
	classifier_ref has a value which is a string
	accuracy has a value which is a float


=end text

=item Description

build_classifier: build_classifier

requried params:

=back

=cut

 sub build_classifier
{
    my($self, @args) = @_;

# Authentication: required

    if ((my $n = @args) != 1)
    {
	Bio::KBase::Exceptions::ArgumentValidationError->throw(error =>
							       "Invalid argument count for function build_classifier (received $n, expecting 1)");
    }
    {
	my($params) = @args;

	my @_bad_arguments;
        (ref($params) eq 'HASH') or push(@_bad_arguments, "Invalid type for argument 1 \"params\" (value was \"$params\")");
        if (@_bad_arguments) {
	    my $msg = "Invalid arguments passed to build_classifier:\n" . join("", map { "\t$_\n" } @_bad_arguments);
	    Bio::KBase::Exceptions::ArgumentValidationError->throw(error => $msg,
								   method_name => 'build_classifier');
	}
    }

    my $url = $self->{url};
    my $result = $self->{client}->call($url, $self->{headers}, {
	    method => "kb_genomeclassification.build_classifier",
	    params => \@args,
    });
    if ($result) {
	if ($result->is_error) {
	    Bio::KBase::Exceptions::JSONRPC->throw(error => $result->error_message,
					       code => $result->content->{error}->{code},
					       method_name => 'build_classifier',
					       data => $result->content->{error}->{error} # JSON::RPC::ReturnObject only supports JSONRPC 1.1 or 1.O
					      );
	} else {
	    return wantarray ? @{$result->result} : $result->result->[0];
	}
    } else {
        Bio::KBase::Exceptions::HTTP->throw(error => "Error invoking method build_classifier",
					    status_line => $self->{client}->status_line,
					    method_name => 'build_classifier',
				       );
    }
}
 


=head2 predict_phenotype

  $output = $obj->predict_phenotype($params)

=over 4

=item Parameter and return types

=begin html

<pre>
$params is a kb_genomeclassification.ClassifierPredictionInput
$output is a kb_genomeclassification.ClassifierPredictionOutput
ClassifierPredictionInput is a reference to a hash where the following keys are defined:
	workspace has a value which is a string
	categorizer_name has a value which is a string
	description has a value which is a string
	file_path has a value which is a string
	annotate has a value which is an int
ClassifierPredictionOutput is a reference to a hash where the following keys are defined:
	prediction_set has a value which is a reference to a hash where the key is a string and the value is a kb_genomeclassification.PredictedPhenotypeOut
	report_name has a value which is a string
	report_ref has a value which is a string
PredictedPhenotypeOut is a reference to a hash where the following keys are defined:
	prediction_probabilities has a value which is a float
	phenotype has a value which is a string
	genome_name has a value which is a string
	genome_ref has a value which is a string

</pre>

=end html

=begin text

$params is a kb_genomeclassification.ClassifierPredictionInput
$output is a kb_genomeclassification.ClassifierPredictionOutput
ClassifierPredictionInput is a reference to a hash where the following keys are defined:
	workspace has a value which is a string
	categorizer_name has a value which is a string
	description has a value which is a string
	file_path has a value which is a string
	annotate has a value which is an int
ClassifierPredictionOutput is a reference to a hash where the following keys are defined:
	prediction_set has a value which is a reference to a hash where the key is a string and the value is a kb_genomeclassification.PredictedPhenotypeOut
	report_name has a value which is a string
	report_ref has a value which is a string
PredictedPhenotypeOut is a reference to a hash where the following keys are defined:
	prediction_probabilities has a value which is a float
	phenotype has a value which is a string
	genome_name has a value which is a string
	genome_ref has a value which is a string


=end text

=item Description



=back

=cut

 sub predict_phenotype
{
    my($self, @args) = @_;

# Authentication: required

    if ((my $n = @args) != 1)
    {
	Bio::KBase::Exceptions::ArgumentValidationError->throw(error =>
							       "Invalid argument count for function predict_phenotype (received $n, expecting 1)");
    }
    {
	my($params) = @args;

	my @_bad_arguments;
        (ref($params) eq 'HASH') or push(@_bad_arguments, "Invalid type for argument 1 \"params\" (value was \"$params\")");
        if (@_bad_arguments) {
	    my $msg = "Invalid arguments passed to predict_phenotype:\n" . join("", map { "\t$_\n" } @_bad_arguments);
	    Bio::KBase::Exceptions::ArgumentValidationError->throw(error => $msg,
								   method_name => 'predict_phenotype');
	}
    }

    my $url = $self->{url};
    my $result = $self->{client}->call($url, $self->{headers}, {
	    method => "kb_genomeclassification.predict_phenotype",
	    params => \@args,
    });
    if ($result) {
	if ($result->is_error) {
	    Bio::KBase::Exceptions::JSONRPC->throw(error => $result->error_message,
					       code => $result->content->{error}->{code},
					       method_name => 'predict_phenotype',
					       data => $result->content->{error}->{error} # JSON::RPC::ReturnObject only supports JSONRPC 1.1 or 1.O
					      );
	} else {
	    return wantarray ? @{$result->result} : $result->result->[0];
	}
    } else {
        Bio::KBase::Exceptions::HTTP->throw(error => "Error invoking method predict_phenotype",
					    status_line => $self->{client}->status_line,
					    method_name => 'predict_phenotype',
				       );
    }
}
 


=head2 upload_trainingset

  $output = $obj->upload_trainingset($params)

=over 4

=item Parameter and return types

=begin html

<pre>
$params is a kb_genomeclassification.UploadTrainingSetInput
$output is a kb_genomeclassification.UploadTrainingSetOut
UploadTrainingSetInput is a reference to a hash where the following keys are defined:
	phenotype has a value which is a string
	workspace has a value which is a string
	workspace_id has a value which is a string
	description has a value which is a string
	training_set_name has a value which is a string
	file_path has a value which is a string
	annotate has a value which is an int
UploadTrainingSetOut is a reference to a hash where the following keys are defined:
	classifier_training_set has a value which is a reference to a hash where the key is a string and the value is a kb_genomeclassification.ClassifierTrainingSetOut
	report_name has a value which is a string
	report_ref has a value which is a string
ClassifierTrainingSetOut is a reference to a hash where the following keys are defined:
	phenotype has a value which is a string
	genome_name has a value which is a string
	genome_ref has a value which is a string
	references has a value which is a reference to a list where each element is a string
	evidence_types has a value which is a reference to a list where each element is a string

</pre>

=end html

=begin text

$params is a kb_genomeclassification.UploadTrainingSetInput
$output is a kb_genomeclassification.UploadTrainingSetOut
UploadTrainingSetInput is a reference to a hash where the following keys are defined:
	phenotype has a value which is a string
	workspace has a value which is a string
	workspace_id has a value which is a string
	description has a value which is a string
	training_set_name has a value which is a string
	file_path has a value which is a string
	annotate has a value which is an int
UploadTrainingSetOut is a reference to a hash where the following keys are defined:
	classifier_training_set has a value which is a reference to a hash where the key is a string and the value is a kb_genomeclassification.ClassifierTrainingSetOut
	report_name has a value which is a string
	report_ref has a value which is a string
ClassifierTrainingSetOut is a reference to a hash where the following keys are defined:
	phenotype has a value which is a string
	genome_name has a value which is a string
	genome_ref has a value which is a string
	references has a value which is a reference to a list where each element is a string
	evidence_types has a value which is a reference to a list where each element is a string


=end text

=item Description



=back

=cut

 sub upload_trainingset
{
    my($self, @args) = @_;

# Authentication: required

    if ((my $n = @args) != 1)
    {
	Bio::KBase::Exceptions::ArgumentValidationError->throw(error =>
							       "Invalid argument count for function upload_trainingset (received $n, expecting 1)");
    }
    {
	my($params) = @args;

	my @_bad_arguments;
        (ref($params) eq 'HASH') or push(@_bad_arguments, "Invalid type for argument 1 \"params\" (value was \"$params\")");
        if (@_bad_arguments) {
	    my $msg = "Invalid arguments passed to upload_trainingset:\n" . join("", map { "\t$_\n" } @_bad_arguments);
	    Bio::KBase::Exceptions::ArgumentValidationError->throw(error => $msg,
								   method_name => 'upload_trainingset');
	}
    }

    my $url = $self->{url};
    my $result = $self->{client}->call($url, $self->{headers}, {
	    method => "kb_genomeclassification.upload_trainingset",
	    params => \@args,
    });
    if ($result) {
	if ($result->is_error) {
	    Bio::KBase::Exceptions::JSONRPC->throw(error => $result->error_message,
					       code => $result->content->{error}->{code},
					       method_name => 'upload_trainingset',
					       data => $result->content->{error}->{error} # JSON::RPC::ReturnObject only supports JSONRPC 1.1 or 1.O
					      );
	} else {
	    return wantarray ? @{$result->result} : $result->result->[0];
	}
    } else {
        Bio::KBase::Exceptions::HTTP->throw(error => "Error invoking method upload_trainingset",
					    status_line => $self->{client}->status_line,
					    method_name => 'upload_trainingset',
				       );
    }
}
 


=head2 rast_annotate_trainingset

  $output = $obj->rast_annotate_trainingset($params)

=over 4

=item Parameter and return types

=begin html

<pre>
$params is a kb_genomeclassification.RastAnnotateTrainingSetInput
$output is a kb_genomeclassification.RastAnnotateTrainingSetOutput
RastAnnotateTrainingSetInput is a reference to a hash where the following keys are defined:
	classifier_training_set has a value which is a reference to a hash where the key is a string and the value is a kb_genomeclassification.ClassifierTrainingSetOut
	workspace has a value which is a string
	make_genome_set has a value which is an int
ClassifierTrainingSetOut is a reference to a hash where the following keys are defined:
	phenotype has a value which is a string
	genome_name has a value which is a string
	genome_ref has a value which is a string
	references has a value which is a reference to a list where each element is a string
	evidence_types has a value which is a reference to a list where each element is a string
RastAnnotateTrainingSetOutput is a reference to a hash where the following keys are defined:
	classifier_training_set has a value which is a reference to a hash where the key is a string and the value is a kb_genomeclassification.ClassifierTrainingSetOut
	report_name has a value which is a string
	report_ref has a value which is a string

</pre>

=end html

=begin text

$params is a kb_genomeclassification.RastAnnotateTrainingSetInput
$output is a kb_genomeclassification.RastAnnotateTrainingSetOutput
RastAnnotateTrainingSetInput is a reference to a hash where the following keys are defined:
	classifier_training_set has a value which is a reference to a hash where the key is a string and the value is a kb_genomeclassification.ClassifierTrainingSetOut
	workspace has a value which is a string
	make_genome_set has a value which is an int
ClassifierTrainingSetOut is a reference to a hash where the following keys are defined:
	phenotype has a value which is a string
	genome_name has a value which is a string
	genome_ref has a value which is a string
	references has a value which is a reference to a list where each element is a string
	evidence_types has a value which is a reference to a list where each element is a string
RastAnnotateTrainingSetOutput is a reference to a hash where the following keys are defined:
	classifier_training_set has a value which is a reference to a hash where the key is a string and the value is a kb_genomeclassification.ClassifierTrainingSetOut
	report_name has a value which is a string
	report_ref has a value which is a string


=end text

=item Description



=back

=cut

 sub rast_annotate_trainingset
{
    my($self, @args) = @_;

# Authentication: required

    if ((my $n = @args) != 1)
    {
	Bio::KBase::Exceptions::ArgumentValidationError->throw(error =>
							       "Invalid argument count for function rast_annotate_trainingset (received $n, expecting 1)");
    }
    {
	my($params) = @args;

	my @_bad_arguments;
        (ref($params) eq 'HASH') or push(@_bad_arguments, "Invalid type for argument 1 \"params\" (value was \"$params\")");
        if (@_bad_arguments) {
	    my $msg = "Invalid arguments passed to rast_annotate_trainingset:\n" . join("", map { "\t$_\n" } @_bad_arguments);
	    Bio::KBase::Exceptions::ArgumentValidationError->throw(error => $msg,
								   method_name => 'rast_annotate_trainingset');
	}
    }

    my $url = $self->{url};
    my $result = $self->{client}->call($url, $self->{headers}, {
	    method => "kb_genomeclassification.rast_annotate_trainingset",
	    params => \@args,
    });
    if ($result) {
	if ($result->is_error) {
	    Bio::KBase::Exceptions::JSONRPC->throw(error => $result->error_message,
					       code => $result->content->{error}->{code},
					       method_name => 'rast_annotate_trainingset',
					       data => $result->content->{error}->{error} # JSON::RPC::ReturnObject only supports JSONRPC 1.1 or 1.O
					      );
	} else {
	    return wantarray ? @{$result->result} : $result->result->[0];
	}
    } else {
        Bio::KBase::Exceptions::HTTP->throw(error => "Error invoking method rast_annotate_trainingset",
					    status_line => $self->{client}->status_line,
					    method_name => 'rast_annotate_trainingset',
				       );
    }
}
 
  
sub status
{
    my($self, @args) = @_;
    if ((my $n = @args) != 0) {
        Bio::KBase::Exceptions::ArgumentValidationError->throw(error =>
                                   "Invalid argument count for function status (received $n, expecting 0)");
    }
    my $url = $self->{url};
    my $result = $self->{client}->call($url, $self->{headers}, {
        method => "kb_genomeclassification.status",
        params => \@args,
    });
    if ($result) {
        if ($result->is_error) {
            Bio::KBase::Exceptions::JSONRPC->throw(error => $result->error_message,
                           code => $result->content->{error}->{code},
                           method_name => 'status',
                           data => $result->content->{error}->{error} # JSON::RPC::ReturnObject only supports JSONRPC 1.1 or 1.O
                          );
        } else {
            return wantarray ? @{$result->result} : $result->result->[0];
        }
    } else {
        Bio::KBase::Exceptions::HTTP->throw(error => "Error invoking method status",
                        status_line => $self->{client}->status_line,
                        method_name => 'status',
                       );
    }
}
   

sub version {
    my ($self) = @_;
    my $result = $self->{client}->call($self->{url}, $self->{headers}, {
        method => "kb_genomeclassification.version",
        params => [],
    });
    if ($result) {
        if ($result->is_error) {
            Bio::KBase::Exceptions::JSONRPC->throw(
                error => $result->error_message,
                code => $result->content->{code},
                method_name => 'rast_annotate_trainingset',
            );
        } else {
            return wantarray ? @{$result->result} : $result->result->[0];
        }
    } else {
        Bio::KBase::Exceptions::HTTP->throw(
            error => "Error invoking method rast_annotate_trainingset",
            status_line => $self->{client}->status_line,
            method_name => 'rast_annotate_trainingset',
        );
    }
}

sub _validate_version {
    my ($self) = @_;
    my $svr_version = $self->version();
    my $client_version = $VERSION;
    my ($cMajor, $cMinor) = split(/\./, $client_version);
    my ($sMajor, $sMinor) = split(/\./, $svr_version);
    if ($sMajor != $cMajor) {
        Bio::KBase::Exceptions::ClientServerIncompatible->throw(
            error => "Major version numbers differ.",
            server_version => $svr_version,
            client_version => $client_version
        );
    }
    if ($sMinor < $cMinor) {
        Bio::KBase::Exceptions::ClientServerIncompatible->throw(
            error => "Client minor version greater than Server minor version.",
            server_version => $svr_version,
            client_version => $client_version
        );
    }
    if ($sMinor > $cMinor) {
        warn "New client version available for kb_genomeclassification::kb_genomeclassificationClient\n";
    }
    if ($sMajor == 0) {
        warn "kb_genomeclassification::kb_genomeclassificationClient version is $svr_version. API subject to change.\n";
    }
}

=head1 TYPES



=head2 boolean

=over 4



=item Description

"True" or "False"


=item Definition

=begin html

<pre>
a string
</pre>

=end html

=begin text

a string

=end text

=back



=head2 ClassifierTrainingSet

=over 4



=item Definition

=begin html

<pre>
a reference to a hash where the following keys are defined:
phenotype has a value which is a string
genome_name has a value which is a string

</pre>

=end html

=begin text

a reference to a hash where the following keys are defined:
phenotype has a value which is a string
genome_name has a value which is a string


=end text

=back



=head2 LogisticRegressionOptions

=over 4



=item Definition

=begin html

<pre>
a reference to a hash where the following keys are defined:
penalty has a value which is a string
dual has a value which is a kb_genomeclassification.boolean
lr_tolerance has a value which is a float
lr_C has a value which is a float
fit_intercept has a value which is a kb_genomeclassification.boolean
intercept_scaling has a value which is a float
lr_class_weight has a value which is a string
lr_random_state has a value which is an int
lr_solver has a value which is a string
lr_max_iter has a value which is an int
multi_class has a value which is a string
lr_verbose has a value which is a kb_genomeclassification.boolean
lr_warm_start has a value which is an int
lr_n_jobs has a value which is an int

</pre>

=end html

=begin text

a reference to a hash where the following keys are defined:
penalty has a value which is a string
dual has a value which is a kb_genomeclassification.boolean
lr_tolerance has a value which is a float
lr_C has a value which is a float
fit_intercept has a value which is a kb_genomeclassification.boolean
intercept_scaling has a value which is a float
lr_class_weight has a value which is a string
lr_random_state has a value which is an int
lr_solver has a value which is a string
lr_max_iter has a value which is an int
multi_class has a value which is a string
lr_verbose has a value which is a kb_genomeclassification.boolean
lr_warm_start has a value which is an int
lr_n_jobs has a value which is an int


=end text

=back



=head2 DecisionTreeClassifierOptions

=over 4



=item Definition

=begin html

<pre>
a reference to a hash where the following keys are defined:
criterion has a value which is a string
splitter has a value which is a string
max_depth has a value which is an int
min_samples_split has a value which is an int
min_samples_leaf has a value which is an int
min_weight_fraction_leaf has a value which is a float
max_features has a value which is a string
dt_random_state has a value which is an int
max_leaf_nodes has a value which is an int
min_impurity_decrease has a value which is a float
dt_class_weight has a value which is a string
presort has a value which is a string

</pre>

=end html

=begin text

a reference to a hash where the following keys are defined:
criterion has a value which is a string
splitter has a value which is a string
max_depth has a value which is an int
min_samples_split has a value which is an int
min_samples_leaf has a value which is an int
min_weight_fraction_leaf has a value which is a float
max_features has a value which is a string
dt_random_state has a value which is an int
max_leaf_nodes has a value which is an int
min_impurity_decrease has a value which is a float
dt_class_weight has a value which is a string
presort has a value which is a string


=end text

=back



=head2 GaussianNBOptions

=over 4



=item Definition

=begin html

<pre>
a reference to a hash where the following keys are defined:
priors has a value which is a string

</pre>

=end html

=begin text

a reference to a hash where the following keys are defined:
priors has a value which is a string


=end text

=back



=head2 KNearestNeighborsOptions

=over 4



=item Definition

=begin html

<pre>
a reference to a hash where the following keys are defined:
n_neighbors has a value which is an int
weights has a value which is a string
algorithm has a value which is a string
leaf_size has a value which is an int
p has a value which is an int
metric has a value which is a string
metric_params has a value which is a string
knn_n_jobs has a value which is an int

</pre>

=end html

=begin text

a reference to a hash where the following keys are defined:
n_neighbors has a value which is an int
weights has a value which is a string
algorithm has a value which is a string
leaf_size has a value which is an int
p has a value which is an int
metric has a value which is a string
metric_params has a value which is a string
knn_n_jobs has a value which is an int


=end text

=back



=head2 SupportVectorMachineOptions

=over 4



=item Definition

=begin html

<pre>
a reference to a hash where the following keys are defined:
svm_C has a value which is a float
kernel has a value which is a string
degree has a value which is an int
gamma has a value which is a string
coef0 has a value which is a float
probability has a value which is a kb_genomeclassification.boolean
shrinking has a value which is a kb_genomeclassification.boolean
svm_tolerance has a value which is a float
cache_size has a value which is a float
svm_class_weight has a value which is a string
svm_verbose has a value which is a kb_genomeclassification.boolean
svm_max_iter has a value which is an int
decision_function_shape has a value which is a string
svm_random_state has a value which is an int

</pre>

=end html

=begin text

a reference to a hash where the following keys are defined:
svm_C has a value which is a float
kernel has a value which is a string
degree has a value which is an int
gamma has a value which is a string
coef0 has a value which is a float
probability has a value which is a kb_genomeclassification.boolean
shrinking has a value which is a kb_genomeclassification.boolean
svm_tolerance has a value which is a float
cache_size has a value which is a float
svm_class_weight has a value which is a string
svm_verbose has a value which is a kb_genomeclassification.boolean
svm_max_iter has a value which is an int
decision_function_shape has a value which is a string
svm_random_state has a value which is an int


=end text

=back



=head2 NeuralNetworkOptions

=over 4



=item Definition

=begin html

<pre>
a reference to a hash where the following keys are defined:
hidden_layer_sizes has a value which is a string
activation has a value which is a string
mlp_solver has a value which is a string
alpha has a value which is a float
batch_size has a value which is a string
learning_rate has a value which is a string
learning_rate_init has a value which is a float
power_t has a value which is a float
mlp_max_iter has a value which is an int
shuffle has a value which is a kb_genomeclassification.boolean
mlp_random_state has a value which is an int
mlp_tolerance has a value which is a float
mlp_verbose has a value which is a kb_genomeclassification.boolean
mlp_warm_start has a value which is a kb_genomeclassification.boolean
momentum has a value which is a float
nesterovs_momentum has a value which is a kb_genomeclassification.boolean
early_stopping has a value which is a kb_genomeclassification.boolean
validation_fraction has a value which is a float
beta_1 has a value which is a float
beta_2 has a value which is a float
epsilon has a value which is a float

</pre>

=end html

=begin text

a reference to a hash where the following keys are defined:
hidden_layer_sizes has a value which is a string
activation has a value which is a string
mlp_solver has a value which is a string
alpha has a value which is a float
batch_size has a value which is a string
learning_rate has a value which is a string
learning_rate_init has a value which is a float
power_t has a value which is a float
mlp_max_iter has a value which is an int
shuffle has a value which is a kb_genomeclassification.boolean
mlp_random_state has a value which is an int
mlp_tolerance has a value which is a float
mlp_verbose has a value which is a kb_genomeclassification.boolean
mlp_warm_start has a value which is a kb_genomeclassification.boolean
momentum has a value which is a float
nesterovs_momentum has a value which is a kb_genomeclassification.boolean
early_stopping has a value which is a kb_genomeclassification.boolean
validation_fraction has a value which is a float
beta_1 has a value which is a float
beta_2 has a value which is a float
epsilon has a value which is a float


=end text

=back



=head2 EnsembleModelOptions

=over 4



=item Definition

=begin html

<pre>
a reference to a hash where the following keys are defined:
k_nearest_neighbors_box has a value which is an int
gaussian_nb_box has a value which is an int
logistic_regression_box has a value which is an int
decision_tree_classifier_box has a value which is an int
support_vector_machine_box has a value which is an int
neural_network_box has a value which is an int
voting has a value which is a string
en_weights has a value which is a string
en_n_jobs has a value which is an int
flatten_transform has a value which is a kb_genomeclassification.boolean

</pre>

=end html

=begin text

a reference to a hash where the following keys are defined:
k_nearest_neighbors_box has a value which is an int
gaussian_nb_box has a value which is an int
logistic_regression_box has a value which is an int
decision_tree_classifier_box has a value which is an int
support_vector_machine_box has a value which is an int
neural_network_box has a value which is an int
voting has a value which is a string
en_weights has a value which is a string
en_n_jobs has a value which is an int
flatten_transform has a value which is a kb_genomeclassification.boolean


=end text

=back



=head2 BuildClassifierInput

=over 4



=item Definition

=begin html

<pre>
a reference to a hash where the following keys are defined:
genome_attribute has a value which is a string
workspace has a value which is a string
training_set_name has a value which is a string
classifier_training_set has a value which is a reference to a hash where the key is a string and the value is a kb_genomeclassification.ClassifierTrainingSet
classifier_object_name has a value which is a string
description has a value which is a string
classifier_to_run has a value which is a string
logistic_regression has a value which is a kb_genomeclassification.LogisticRegressionOptions
decision_tree_classifier has a value which is a kb_genomeclassification.DecisionTreeClassifierOptions
gaussian_nb has a value which is a kb_genomeclassification.GaussianNBOptions
k_nearest_neighbors has a value which is a kb_genomeclassification.KNearestNeighborsOptions
support_vector_machine has a value which is a kb_genomeclassification.SupportVectorMachineOptions
neural_network has a value which is a kb_genomeclassification.NeuralNetworkOptions
ensemble_model has a value which is a kb_genomeclassification.EnsembleModelOptions

</pre>

=end html

=begin text

a reference to a hash where the following keys are defined:
genome_attribute has a value which is a string
workspace has a value which is a string
training_set_name has a value which is a string
classifier_training_set has a value which is a reference to a hash where the key is a string and the value is a kb_genomeclassification.ClassifierTrainingSet
classifier_object_name has a value which is a string
description has a value which is a string
classifier_to_run has a value which is a string
logistic_regression has a value which is a kb_genomeclassification.LogisticRegressionOptions
decision_tree_classifier has a value which is a kb_genomeclassification.DecisionTreeClassifierOptions
gaussian_nb has a value which is a kb_genomeclassification.GaussianNBOptions
k_nearest_neighbors has a value which is a kb_genomeclassification.KNearestNeighborsOptions
support_vector_machine has a value which is a kb_genomeclassification.SupportVectorMachineOptions
neural_network has a value which is a kb_genomeclassification.NeuralNetworkOptions
ensemble_model has a value which is a kb_genomeclassification.EnsembleModelOptions


=end text

=back



=head2 classifierInfo

=over 4



=item Definition

=begin html

<pre>
a reference to a hash where the following keys are defined:
classifier_name has a value which is a string
classifier_ref has a value which is a string
accuracy has a value which is a float

</pre>

=end html

=begin text

a reference to a hash where the following keys are defined:
classifier_name has a value which is a string
classifier_ref has a value which is a string
accuracy has a value which is a float


=end text

=back



=head2 ClassifierOut

=over 4



=item Definition

=begin html

<pre>
a reference to a hash where the following keys are defined:
classifier_info has a value which is a reference to a list where each element is a kb_genomeclassification.classifierInfo
report_name has a value which is a string
report_ref has a value which is a string

</pre>

=end html

=begin text

a reference to a hash where the following keys are defined:
classifier_info has a value which is a reference to a list where each element is a kb_genomeclassification.classifierInfo
report_name has a value which is a string
report_ref has a value which is a string


=end text

=back



=head2 ClassifierPredictionInput

=over 4



=item Definition

=begin html

<pre>
a reference to a hash where the following keys are defined:
workspace has a value which is a string
categorizer_name has a value which is a string
description has a value which is a string
file_path has a value which is a string
annotate has a value which is an int

</pre>

=end html

=begin text

a reference to a hash where the following keys are defined:
workspace has a value which is a string
categorizer_name has a value which is a string
description has a value which is a string
file_path has a value which is a string
annotate has a value which is an int


=end text

=back



=head2 PredictedPhenotypeOut

=over 4



=item Definition

=begin html

<pre>
a reference to a hash where the following keys are defined:
prediction_probabilities has a value which is a float
phenotype has a value which is a string
genome_name has a value which is a string
genome_ref has a value which is a string

</pre>

=end html

=begin text

a reference to a hash where the following keys are defined:
prediction_probabilities has a value which is a float
phenotype has a value which is a string
genome_name has a value which is a string
genome_ref has a value which is a string


=end text

=back



=head2 ClassifierPredictionOutput

=over 4



=item Definition

=begin html

<pre>
a reference to a hash where the following keys are defined:
prediction_set has a value which is a reference to a hash where the key is a string and the value is a kb_genomeclassification.PredictedPhenotypeOut
report_name has a value which is a string
report_ref has a value which is a string

</pre>

=end html

=begin text

a reference to a hash where the following keys are defined:
prediction_set has a value which is a reference to a hash where the key is a string and the value is a kb_genomeclassification.PredictedPhenotypeOut
report_name has a value which is a string
report_ref has a value which is a string


=end text

=back



=head2 UploadTrainingSetInput

=over 4



=item Definition

=begin html

<pre>
a reference to a hash where the following keys are defined:
phenotype has a value which is a string
workspace has a value which is a string
workspace_id has a value which is a string
description has a value which is a string
training_set_name has a value which is a string
file_path has a value which is a string
annotate has a value which is an int

</pre>

=end html

=begin text

a reference to a hash where the following keys are defined:
phenotype has a value which is a string
workspace has a value which is a string
workspace_id has a value which is a string
description has a value which is a string
training_set_name has a value which is a string
file_path has a value which is a string
annotate has a value which is an int


=end text

=back



=head2 ClassifierTrainingSetOut

=over 4



=item Definition

=begin html

<pre>
a reference to a hash where the following keys are defined:
phenotype has a value which is a string
genome_name has a value which is a string
genome_ref has a value which is a string
references has a value which is a reference to a list where each element is a string
evidence_types has a value which is a reference to a list where each element is a string

</pre>

=end html

=begin text

a reference to a hash where the following keys are defined:
phenotype has a value which is a string
genome_name has a value which is a string
genome_ref has a value which is a string
references has a value which is a reference to a list where each element is a string
evidence_types has a value which is a reference to a list where each element is a string


=end text

=back



=head2 UploadTrainingSetOut

=over 4



=item Definition

=begin html

<pre>
a reference to a hash where the following keys are defined:
classifier_training_set has a value which is a reference to a hash where the key is a string and the value is a kb_genomeclassification.ClassifierTrainingSetOut
report_name has a value which is a string
report_ref has a value which is a string

</pre>

=end html

=begin text

a reference to a hash where the following keys are defined:
classifier_training_set has a value which is a reference to a hash where the key is a string and the value is a kb_genomeclassification.ClassifierTrainingSetOut
report_name has a value which is a string
report_ref has a value which is a string


=end text

=back



=head2 RastAnnotateTrainingSetInput

=over 4



=item Definition

=begin html

<pre>
a reference to a hash where the following keys are defined:
classifier_training_set has a value which is a reference to a hash where the key is a string and the value is a kb_genomeclassification.ClassifierTrainingSetOut
workspace has a value which is a string
make_genome_set has a value which is an int

</pre>

=end html

=begin text

a reference to a hash where the following keys are defined:
classifier_training_set has a value which is a reference to a hash where the key is a string and the value is a kb_genomeclassification.ClassifierTrainingSetOut
workspace has a value which is a string
make_genome_set has a value which is an int


=end text

=back



=head2 RastAnnotateTrainingSetOutput

=over 4



=item Definition

=begin html

<pre>
a reference to a hash where the following keys are defined:
classifier_training_set has a value which is a reference to a hash where the key is a string and the value is a kb_genomeclassification.ClassifierTrainingSetOut
report_name has a value which is a string
report_ref has a value which is a string

</pre>

=end html

=begin text

a reference to a hash where the following keys are defined:
classifier_training_set has a value which is a reference to a hash where the key is a string and the value is a kb_genomeclassification.ClassifierTrainingSetOut
report_name has a value which is a string
report_ref has a value which is a string


=end text

=back



=cut

package kb_genomeclassification::kb_genomeclassificationClient::RpcClient;
use base 'JSON::RPC::Client';
use POSIX;
use strict;

#
# Override JSON::RPC::Client::call because it doesn't handle error returns properly.
#

sub call {
    my ($self, $uri, $headers, $obj) = @_;
    my $result;


    {
	if ($uri =~ /\?/) {
	    $result = $self->_get($uri);
	}
	else {
	    Carp::croak "not hashref." unless (ref $obj eq 'HASH');
	    $result = $self->_post($uri, $headers, $obj);
	}

    }

    my $service = $obj->{method} =~ /^system\./ if ( $obj );

    $self->status_line($result->status_line);

    if ($result->is_success) {

        return unless($result->content); # notification?

        if ($service) {
            return JSON::RPC::ServiceObject->new($result, $self->json);
        }

        return JSON::RPC::ReturnObject->new($result, $self->json);
    }
    elsif ($result->content_type eq 'application/json')
    {
        return JSON::RPC::ReturnObject->new($result, $self->json);
    }
    else {
        return;
    }
}


sub _post {
    my ($self, $uri, $headers, $obj) = @_;
    my $json = $self->json;

    $obj->{version} ||= $self->{version} || '1.1';

    if ($obj->{version} eq '1.0') {
        delete $obj->{version};
        if (exists $obj->{id}) {
            $self->id($obj->{id}) if ($obj->{id}); # if undef, it is notification.
        }
        else {
            $obj->{id} = $self->id || ($self->id('JSON::RPC::Client'));
        }
    }
    else {
        # $obj->{id} = $self->id if (defined $self->id);
	# Assign a random number to the id if one hasn't been set
	$obj->{id} = (defined $self->id) ? $self->id : substr(rand(),2);
    }

    my $content = $json->encode($obj);

    $self->ua->post(
        $uri,
        Content_Type   => $self->{content_type},
        Content        => $content,
        Accept         => 'application/json',
	@$headers,
	($self->{token} ? (Authorization => $self->{token}) : ()),
    );
}



1;
