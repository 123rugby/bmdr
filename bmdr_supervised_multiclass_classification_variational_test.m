% Mehmet Gonen (mehmet.gonen@gmail.com)

function prediction = bmdr_supervised_multiclass_classification_variational_test(X, state)
    rand('state', state.parameters.seed); %#ok<RAND>
    randn('state', state.parameters.seed); %#ok<RAND>

    V = length(state.Q);
    R = size(state.bW.mean, 1) - 1;
    N = zeros(V, 1);
    for o = 1:V
        N(o) = size(X{o}, 2);
    end
    K = size(state.bW.mean, 2);

    for o = 1:V
        prediction.Z{o}.mean = zeros(R, N(o));
        prediction.Z{o}.covariance = zeros(R, N(o));
        for s = 1:R
            prediction.Z{o}.mean(s, :) = state.Q{o}.mean(:, s)' * X{o};
            prediction.Z{o}.covariance(s, :) = state.parameters.sigmaz^2 + diag(X{o}' * state.Q{o}.covariance(:, :, s) * X{o});
        end
    end

    prediction.P = cell(1, V);
    u = randn(1, 1, state.parameters.sample);
    for o = 1:V
        T.mean = zeros(K, N(o));
        T.covariance = zeros(K, N(o));
        for c = 1:K
            T.mean(c, :) = state.bW.mean(:, c)' * [ones(1, N(o)); prediction.Z{o}.mean];
            T.covariance(c, :) = 1 + diag([ones(1, N(o)); prediction.Z{o}.mean]' * state.bW.covariance(:, :, c) * [ones(1, N(o)); prediction.Z{o}.mean]);
        end

        prediction.P{o} = zeros(K, N(o));
        for c = 1:K
            A = repmat(u, [K, N(o), 1]) .* repmat(T.covariance(c, :), [K, 1, state.parameters.sample]) + repmat(T.mean(c, :), [K, 1, state.parameters.sample]) - repmat(T.mean, [1, 1, state.parameters.sample]);
            A = A ./ repmat(T.covariance, [1, 1, state.parameters.sample]);
            A(c, :, :) = [];
            prediction.P{o}(c, :) = mean(prod(normcdf(A), 3), 1);
        end
        prediction.P{o} = prediction.P{o} ./ repmat(sum(prediction.P{o}, 1), K, 1);
    end
end