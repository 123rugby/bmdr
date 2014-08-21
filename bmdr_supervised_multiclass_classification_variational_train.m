% Mehmet Gonen (mehmet.gonen@gmail.com)

function state = bmdr_supervised_multiclass_classification_variational_train(X, y, parameters)
    rand('state', parameters.seed); %#ok<RAND>
    randn('state', parameters.seed); %#ok<RAND>

    R = parameters.R;
    sigmaz = parameters.sigmaz;

    log2pi = log(2 * pi);

    V = length(y);
    D = zeros(V, 1);
    N = zeros(V, 1);
    for o = 1:V
        D(o) = size(X{o}, 1);
        N(o) = size(X{o}, 2);
    end
    K = max(y{1});

    Phi = cell(1, V);
    Q = cell(1, V);
    Z = cell(1, V);
    for o = 1:V
        Phi{o}.shape = (parameters.alpha_phi + 0.5) * ones(D(o), R);
        Phi{o}.scale = parameters.beta_phi * ones(D(o), R);
        Q{o}.mean = randn(D(o), R);
        Q{o}.covariance = repmat(eye(D(o), D(o)), [1, 1, R]);
        Z{o}.mean = randn(R, N(o));
        Z{o}.covariance = eye(R, R);
    end
    lambda.shape = (parameters.alpha_lambda + 0.5) * ones(K, 1);
    lambda.scale = parameters.beta_lambda * ones(K, 1);
    Psi.shape = (parameters.alpha_psi + 0.5) * ones(R, K);
    Psi.scale = parameters.beta_psi * ones(R, K);
    bW.mean = randn(R + 1, K);
    bW.covariance = repmat(eye(R + 1, R + 1), [1, 1, K]);
    T = cell(1, V);
    for o = 1:V
        T{o}.mean = zeros(K, N(o));
        T{o}.covariance = eye(K, K);
    end
    for o = 1:V
        for i = 1:N(o)
            while 1
                T{o}.mean(:, i) = randn(K, 1);
                if T{o}.mean(y{o}(i), i) == max(T{o}.mean(:, i))
                    break;
                end
            end
        end
    end
    normalization = cell(1, V);
    for o = 1:V
        normalization{o} = zeros(N(o), 1);
    end

    XXT = cell(1, V);
    phi_indices = cell(1, V);
    for o = 1:V
        XXT{o} = X{o} * X{o}';
        phi_indices{o} = repmat(logical(eye(D(o), D(o))), [1, 1, R]);
    end
    psi_indices = repmat(logical([zeros(1, R + 1); zeros(R, 1), eye(R, R)]), [1, 1, K]);

    if parameters.progress == 1
        bounds = zeros(parameters.iteration, 1);
    end

    for iter = 1:parameters.iteration
        if mod(iter, 1) == 0
            fprintf(1, '.');
        end
        if mod(iter, 10) == 0
            fprintf(1, ' %5d\n', iter);
        end

        %%%% update Phi
        for o = 1:V
            Phi{o}.scale = 1 ./ (1 / parameters.beta_phi + 0.5 * (Q{o}.mean.^2 + reshape(Q{o}.covariance(phi_indices{o}), D(o), R)));
        end
        %%%% update Q
        for o = 1:V
            for s = 1:R
                Q{o}.covariance(:, :, s) = (diag(Phi{o}.shape(:, s) .* Phi{o}.scale(:, s)) + XXT{o} / sigmaz^2) \ eye(D(o), D(o));
                Q{o}.mean(:, s) = Q{o}.covariance(:, :, s) * (X{o} * Z{o}.mean(s, :)' / sigmaz^2);
            end
        end
        %%%% update Z
        for o = 1:V
            Z{o}.covariance = eye(R, R) / sigmaz^2;
            Z{o}.covariance = Z{o}.covariance + bW.mean(2:R + 1, :) * bW.mean(2:R + 1, :)' + sum(bW.covariance(2:R + 1, 2:R + 1, :), 3);
            Z{o}.covariance = Z{o}.covariance \ eye(R, R);
            Z{o}.mean = Q{o}.mean' * X{o} / sigmaz^2;
            Z{o}.mean = Z{o}.mean + bW.mean(2:end, :) * T{o}.mean - repmat(bW.mean(2:R + 1, :) * bW.mean(1, :)' + sum(bW.covariance(1, 2:R + 1, :), 3)', 1, N(o));
            Z{o}.mean = Z{o}.covariance * Z{o}.mean;
        end
        %%%% update lambda
        lambda.scale = 1 ./ (1 / parameters.beta_lambda + 0.5 * (bW.mean(1, :)'.^2 + squeeze(bW.covariance(1, 1, :))));
        %%%% update Psi
        Psi.scale = 1 ./ (1 / parameters.beta_psi + 0.5 * (bW.mean(2:R + 1, :).^2 + reshape(bW.covariance(psi_indices), R, K)));
        %%%% update b and W
        Z1 = zeros(R, 1);
        ZZT = zeros(R, R);
        Zt = zeros(R + 1, K);
        for o = 1:V
            Z1 = Z1 + Z{o}.mean * ones(N(o), 1);
            ZZT = ZZT + Z{o}.mean * Z{o}.mean' + N(o) * Z{o}.covariance;
            Zt = Zt + [ones(1, N(o)); Z{o}.mean] * T{o}.mean';
        end
        for c = 1:K
            bW.covariance(:, :, c) = [lambda.shape(c, 1) * lambda.scale(c, 1) + sum(N), Z1'; Z1, diag(Psi.shape(:, c) .* Psi.scale(:, c)) + ZZT] \ eye(R + 1, R + 1);
            bW.mean(:, c) = bW.covariance(:, :, c) * Zt(:, c);
        end
        %%%% update T
        for o = 1:V
            T{o}.mean = bW.mean(2:R + 1, :)' * Z{o}.mean + repmat(bW.mean(1, :)', 1, N(o));
            for c = 1:K
                pos = find(y{o} == c);
                [normalization{o}(pos, 1), T{o}.mean(:, pos)] = truncated_normal_mean(T{o}.mean(:, pos), c, parameters.sample, 0);
            end
        end

        lb = 0;
        %%%% p(Phi)
        for o = 1:V
            lb = lb + sum(sum((parameters.alpha_phi - 1) * (psi(Phi{o}.shape) + log(Phi{o}.scale)) - Phi{o}.shape .* Phi{o}.scale / parameters.beta_phi - gammaln(parameters.alpha_phi) - parameters.alpha_phi * log(parameters.beta_phi)));
        end
        %%%% p(Q | Phi)
        for o = 1:V
            for s = 1:R
                lb = lb - 0.5 * Q{o}.mean(:, s)' * diag(Phi{o}.shape(:, s) .* Phi{o}.scale(:, s)) * Q{o}.mean(:, s) - 0.5 * (D(o) * log2pi - sum(log(Phi{o}.shape(:, s) .* Phi{o}.scale(:, s))));
            end
        end
        %%%% p(Z | Q, X)
        for o = 1:V
            lb = lb - 0.5 * (sum(sum(Z{o}.mean .* Z{o}.mean)) + N(o) * sum(diag(Z{o}.covariance))) + sum(sum((Q{o}.mean' * X{o}) .* Z{o}.mean)) - 0.5 * sum(sum(X{o} .* ((Q{o}.mean * Q{o}.mean' + sum(Q{o}.covariance, 3)) * X{o}))) - 0.5 * N(o) * D(o) * (log2pi + 2 * log(sigmaz));
        end
        %%%% p(lambda)
        lb = lb + sum((parameters.alpha_lambda - 1) * (psi(lambda.shape) + log(lambda.scale)) - lambda.shape .* lambda.scale / parameters.beta_lambda - gammaln(parameters.alpha_lambda) - parameters.alpha_lambda * log(parameters.beta_lambda));
        %%%% p(b | lambda)
        lb = lb - 0.5 * bW.mean(1, :) * diag(lambda.shape(:, 1) .* lambda.scale(:, 1)) * bW.mean(1, :)' - 0.5 * (K * log2pi - sum(log(lambda.shape(:, 1) .* lambda.scale(:, 1))));
        %%%% p(Psi)
        lb = lb + sum(sum((parameters.alpha_psi - 1) * (psi(Psi.shape) + log(Psi.scale)) - Psi.shape .* Psi.scale / parameters.beta_psi - gammaln(parameters.alpha_psi) - parameters.alpha_psi * log(parameters.beta_psi)));
        %%%% p(W | Psi)
        for c = 1:K
            lb = lb - 0.5 * bW.mean(2:R + 1, c)' * diag(Psi.shape(:, c) .* Psi.scale(:, c)) * bW.mean(2:R + 1, c) - 0.5 * (R * log2pi - sum(log(Psi.shape(:, c) .* Psi.scale(:, c))));
        end
        %%%% p(T | b, W, Z) p(y | T)
        WWT.mean = bW.mean(2:R + 1, :) * bW.mean(2:R + 1, :)' + sum(bW.covariance(2:R + 1, 2:R + 1, :), 3);
        for o = 1:V
            lb = lb - 0.5 * (sum(sum(T{o}.mean .* T{o}.mean)) + N(o) * K) + sum(bW.mean(1, :) * T{o}.mean) + sum(sum((bW.mean(2:R + 1, :)' * Z{o}.mean) .* T{o}.mean)) - 0.5 * (N(o) * trace(WWT.mean * Z{o}.covariance) + sum(sum(Z{o}.mean .* (WWT.mean * Z{o}.mean)))) - 0.5 * N(o) * (bW.mean(1, :) * bW.mean(1, :)' + sum(bW.covariance(1, 1, :))) - sum(Z{o}.mean' * (bW.mean(2:R + 1, :) * bW.mean(1, :)' + sum(bW.covariance(2:R + 1, 1, :), 3))) - 0.5 * N(o) * K * log2pi;
        end

        %%%% q(Phi)
        for o = 1:V
            lb = lb + sum(sum(Phi{o}.shape + log(Phi{o}.scale) + gammaln(Phi{o}.shape) + (1 - Phi{o}.shape) .* psi(Phi{o}.shape)));
        end
        %%%% q(Q)
        for o = 1:V
            for s = 1:R
                lb = lb + 0.5 * (D(o) * (log2pi + 1) + logdet(Q{o}.covariance(:, :, s)));
            end
        end
        %%%% q(Z)
        for o = 1:V
            lb = lb + 0.5 * N(o) * (R * (log2pi + 1) + logdet(Z{o}.covariance));
        end
        %%%% q(lambda)
        lb = lb + sum(lambda.shape + log(lambda.scale) + gammaln(lambda.shape) + (1 - lambda.shape) .* psi(lambda.shape));
        %%%% q(Psi)
        lb = lb + sum(sum(Psi.shape + log(Psi.scale) + gammaln(Psi.shape) + (1 - Psi.shape) .* psi(Psi.shape)));
        %%%% q(b, W)
        for c = 1:K
            lb = lb + 0.5 * ((R + 1) * (log2pi + 1) + logdet(bW.covariance(:, :, c))); 
        end
        %%%% q(T)
        for o = 1:V
            lb = lb + 0.5 * N(o) * K * (log2pi + 1) + sum(log(normalization{o}));
        end

        bounds(iter) = lb;
    end

    state.Phi = Phi;
    state.Q = Q;
    state.lambda = lambda;
    state.Psi = Psi;
    state.bW = bW;
    if parameters.progress == 1
        state.bounds = bounds;
    end
    state.parameters = parameters;
end

function ld = logdet(Sigma)
    U = chol(Sigma);
    ld = 2 * sum(log(diag(U)));
end

function [normalization, expectation] = truncated_normal_mean(centers, active, S, tube)
    K = size(centers, 1);
    N = size(centers, 2);    
    diff = repmat(centers(active, :), K, 1) - centers - tube;
    u = randn([1, N, S]);
    q = normcdf(repmat(u, [K, 1, 1]) + repmat(diff, [1, 1, S]));
    pr = repmat(prod(q, 1), [K, 1, 1]);
    pr = pr ./ q;
    ind = [1:active - 1, active + 1:K];
    pr(ind, :, :) = pr(ind, :, :) ./ repmat(q(active, :, :), [K - 1, 1, 1]);
    pr(ind, :, :) = pr(ind, :, :) .* normpdf(repmat(u, [K - 1, 1, 1]) + repmat(diff(ind, :), [1, 1, S]));

    normalization = mean(pr(active, :, :), 3);
    expectation = zeros(K, N);
    expectation(ind, :) = centers(ind, :) - repmat(1 ./ normalization, K - 1, 1) .* reshape(mean(pr(ind, :, :), 3), K - 1, N);
    expectation(active, :) = centers(active, :) + sum(centers(ind, :) - expectation(ind, :), 1);
end