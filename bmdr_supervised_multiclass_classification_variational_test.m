% Mehmet Gonen (mehmet.gonen@gmail.com)

function prediction = bmdr_supervised_multiclass_classification_variational_test(X, state)
    rand('state', state.parameters.seed); %#ok<RAND>
    randn('state', state.parameters.seed); %#ok<RAND>

    V = length(state.Q);
    R = size(state.bW.mu, 1) - 1;
    N = zeros(V, 1);
    for o = 1:V
        N(o) = size(X{o}, 2);
    end
    K = size(state.bW.mu, 2);

    for o = 1:V
        prediction.Z{o}.mu = zeros(R, N(o));
        prediction.Z{o}.sigma = zeros(R, N(o));
        for s = 1:R
            prediction.Z{o}.mu(s, :) = state.Q{o}.mu(:, s)' * X{o};
            prediction.Z{o}.sigma(s, :) = state.parameters.sigma_z^2 + diag(X{o}' * state.Q{o}.sigma(:, :, s) * X{o});
        end
    end

    prediction.P = cell(1, V);
    u = randn(1, 1, state.parameters.sample);
    for o = 1:V
        T.mu = zeros(K, N(o));
        T.sigma = zeros(K, N(o));
        for c = 1:K
            T.mu(c, :) = state.bW.mu(:, c)' * [ones(1, N(o)); prediction.Z{o}.mu];
            T.sigma(c, :) = 1 + diag([ones(1, N(o)); prediction.Z{o}.mu]' * state.bW.sigma(:, :, c) * [ones(1, N(o)); prediction.Z{o}.mu]);
        end

        prediction.P{o} = zeros(K, N(o));
        for c = 1:K
            A = repmat(u, [K, N(o), 1]) .* repmat(T.sigma(c, :), [K, 1, state.parameters.sample]) + repmat(T.mu(c, :), [K, 1, state.parameters.sample]) - repmat(T.mu, [1, 1, state.parameters.sample]);
            A = A ./ repmat(T.sigma, [1, 1, state.parameters.sample]);
            A(c, :, :) = [];
            prediction.P{o}(c, :) = mean(prod(normcdf(A), 3), 1);
        end
        prediction.P{o} = prediction.P{o} ./ repmat(sum(prediction.P{o}, 1), K, 1);
    end
end