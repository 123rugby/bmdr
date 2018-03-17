%initalize the parameters of the algorithm
parameters = struct();

%set the hyperparameters of gamma prior used for projection matrices
parameters.alpha_phi = 1;
parameters.beta_phi = 1;

%set the hyperparameters of gamma prior used for bias parameters
parameters.alpha_lambda = 1;
parameters.beta_lambda = 1;

%set the hyperparameters of gamma prior used for weight parameters
parameters.alpha_psi = 1;
parameters.beta_psi = 1;

%%% IMPORTANT %%%
%For gamma priors, you can experiment with three different (alpha, beta) values
%(1, 1) => default priors
%(1e-10, 1e+10) => good for obtaining sparsity
%(1e-10, 1e-10) => good for small sample size problems

%set the number of iterations
parameters.iteration = 200;

%set the subspace dimensionality
parameters.R = 20;

%determine whether you want to calculate and store the lower bound values
parameters.progress = 0;

%set the sample size used to calculate the expectation of truncated normals
parameters.sample = 200;

%set the seed for random number generator used to initalize random variables
parameters.seed = 1606;

%set the standard deviation of projected instances
parameters.sigma_z = 0.1;

%set the number of views
V = ??;

%initialize the data matrix and class labels of each view for training
Xtrain = cell(1, V);
ytrain = cell(1, V);
for o = 1:V
    Xtrain{o} = ??; %should be an D x Ntra matrix containing input data for training samples of view o
    ytrain{o} = ??; %should be an Ntra x 1 matrix containing class labels of view o (contains only 1s, 2s, .., Ks) where K is the number of classes
end

%perform training
state = bmdr_supervised_multiclass_classification_variational_train(Xtrain, ytrain, parameters);

%initialize the data matrix of each view for testing
Xtest = cell(1, V);
for o = 1:V
    Xtest{o} = ??; %should be an D x Ntest matrix containing input data for test samples of view o
end

%perform prediction
prediction = bmdr_supervised_multiclass_classification_variational_test(Xtest, state);

%display the predicted probabilities for each view
for o = 1:V
    display(prediction.P{o});
end
