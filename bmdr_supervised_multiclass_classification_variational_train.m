% Mehmet Gonen (mehmet.gonen@gmail.com)

function state = bmdr_supervised_multiclass_classification_variational_train(X, y, parameters)
    rand('state', parameters.seed); %#ok<RAND>
    randn('state', parameters.seed); %#ok<RAND>

    R = parameters.R;
    sigma_z = parameters.sigma_z;

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
        Phi{o}.alpha = (parameters.alpha_phi + 0.5) * ones(D(o), R);
        Phi{o}.beta = parameters.beta_phi * ones(D(o), R);
        Q{o}.mu = randn(D(o), R);
        Q{o}.sigma = repmat(eye(D(o), D(o)), [1, 1, R]);
        Z{o}.mu = randn(R, N(o));
        Z{o}.sigma = eye(R, R);
    end
    lambda.alpha = (parameters.alpha_lambda + 0.5) * ones(K, 1);
    lambda.beta = parameters.beta_lambda * ones(K, 1);
    Psi.alpha = (parameters.alpha_psi + 0.5) * ones(R, K);
    Psi.beta = parameters.beta_psi * ones(R, K);
    bW.mu = randn(R + 1, K);
    bW.sigma = repmat(eye(R + 1, R + 1), [1, 1, K]);
    T = cell(1, V);
    for o = 1:V
        T{o}.mu = zeros(K, N(o));
        T{o}.sigma = eye(K, K);
    end
    for o = 1:V
        for i = 1:N(o)
            while 1
                T{o}.mu(:, i) = randn(K, 1);
                if T{o}.mu(y{o}(i), i) == max(T{o}.mu(:, i))
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
            Phi{o}.beta = 1 ./ (1 / parameters.beta_phi + 0.5 * (Q{o}.mu.^2 + reshape(Q{o}.sigma(phi_indices{o}), D(o), R)));
        end
        %%%% update Q
        for o = 1:V
            for s = 1:R
                Q{o}.sigma(:, :, s) = (diag(Phi{o}.alpha(:, s) .* Phi{o}.beta(:, s)) + XXT{o} / sigma_z^2) \ eye(D(o), D(o));
                Q{o}.mu(:, s) = Q{o}.sigma(:, :, s) * (X{o} * Z{o}.mu(s, :)' / sigma_z^2);
            end
        end
        %%%% update Z
        for o = 1:V
            Z{o}.sigma = eye(R, R) / sigma_z^2;
            Z{o}.sigma = Z{o}.sigma + bW.mu(2:R + 1, :) * bW.mu(2:R + 1, :)' + sum(bW.sigma(2:R + 1, 2:R + 1, :), 3);
            Z{o}.sigma = Z{o}.sigma \ eye(R, R);
            Z{o}.mu = Q{o}.mu' * X{o} / sigma_z^2;
            Z{o}.mu = Z{o}.mu + bW.mu(2:end, :) * T{o}.mu - repmat(bW.mu(2:R + 1, :) * bW.mu(1, :)' + sum(bW.sigma(1, 2:R + 1, :), 3)', 1, N(o));
            Z{o}.mu = Z{o}.sigma * Z{o}.mu;
        end
        %%%% update lambda
        lambda.beta = 1 ./ (1 / parameters.beta_lambda + 0.5 * (bW.mu(1, :)'.^2 + squeeze(bW.sigma(1, 1, :))));
        %%%% update Psi
        Psi.beta = 1 ./ (1 / parameters.beta_psi + 0.5 * (bW.mu(2:R + 1, :).^2 + reshape(bW.sigma(psi_indices), R, K)));
        %%%% update b and W
        Z1 = zeros(R, 1);
        ZZT = zeros(R, R);
        Zt = zeros(R + 1, K);
        for o = 1:V
            Z1 = Z1 + Z{o}.mu * ones(N(o), 1);
            ZZT = ZZT + Z{o}.mu * Z{o}.mu' + N(o) * Z{o}.sigma;
            Zt = Zt + [ones(1, N(o)); Z{o}.mu] * T{o}.mu';
        end
        for c = 1:K
            bW.sigma(:, :, c) = [lambda.alpha(c, 1) * lambda.beta(c, 1) + sum(N), Z1'; Z1, diag(Psi.alpha(:, c) .* Psi.beta(:, c)) + ZZT] \ eye(R + 1, R + 1);
            bW.mu(:, c) = bW.sigma(:, :, c) * Zt(:, c);
        end
        %%%% update T
        for o = 1:V
            T{o}.mu = bW.mu(2:R + 1, :)' * Z{o}.mu + repmat(bW.mu(1, :)', 1, N(o));
            for c = 1:K
                pos = find(y{o} == c);
                [normalization{o}(pos, 1), T{o}.mu(:, pos)] = truncated_normal_mean(T{o}.mu(:, pos), c, parameters.sample, 0);
            end
        end

        lb = 0;
        %%%% p(Phi)
        for o = 1:V
            lb = lb + sum(sum((parameters.alpha_phi - 1) * (psi(Phi{o}.alpha) + log(Phi{o}.beta)) - Phi{o}.alpha .* Phi{o}.beta / parameters.beta_phi - gammaln(parameters.alpha_phi) - parameters.alpha_phi * log(parameters.beta_phi)));
        end
        %%%% p(Q | Phi)
        for o = 1:V
            for s = 1:R
                lb = lb - 0.5 * Q{o}.mu(:, s)' * diag(Phi{o}.alpha(:, s) .* Phi{o}.beta(:, s)) * Q{o}.mu(:, s) - 0.5 * (D(o) * log2pi - sum(log(Phi{o}.alpha(:, s) .* Phi{o}.beta(:, s))));
            end
        end
        %%%% p(Z | Q, X)
        for o = 1:V
            lb = lb - 0.5 * (sum(sum(Z{o}.mu .* Z{o}.mu)) + N(o) * sum(diag(Z{o}.sigma))) + sum(sum((Q{o}.mu' * X{o}) .* Z{o}.mu)) - 0.5 * sum(sum(X{o} .* ((Q{o}.mu * Q{o}.mu' + sum(Q{o}.sigma, 3)) * X{o}))) - 0.5 * N(o) * D(o) * (log2pi + 2 * log(sigma_z));
        end
        %%%% p(lambda)
        lb = lb + sum((parameters.alpha_lambda - 1) * (psi(lambda.alpha) + log(lambda.beta)) - lambda.alpha .* lambda.beta / parameters.beta_lambda - gammaln(parameters.alpha_lambda) - parameters.alpha_lambda * log(parameters.beta_lambda));
        %%%% p(b | lambda)
        lb = lb - 0.5 * bW.mu(1, :) * diag(lambda.alpha(:, 1) .* lambda.beta(:, 1)) * bW.mu(1, :)' - 0.5 * (K * log2pi - sum(log(lambda.alpha(:, 1) .* lambda.beta(:, 1))));
        %%%% p(Psi)
        lb = lb + sum(sum((parameters.alpha_psi - 1) * (psi(Psi.alpha) + log(Psi.beta)) - Psi.alpha .* Psi.beta / parameters.beta_psi - gammaln(parameters.alpha_psi) - parameters.alpha_psi * log(parameters.beta_psi)));
        %%%% p(W | Psi)
        for c = 1:K
            lb = lb - 0.5 * bW.mu(2:R + 1, c)' * diag(Psi.alpha(:, c) .* Psi.beta(:, c)) * bW.mu(2:R + 1, c) - 0.5 * (R * log2pi - sum(log(Psi.alpha(:, c) .* Psi.beta(:, c))));
        end
        %%%% p(T | b, W, Z) p(y | T)
        WWT.mu = bW.mu(2:R + 1, :) * bW.mu(2:R + 1, :)' + sum(bW.sigma(2:R + 1, 2:R + 1, :), 3);
        for o = 1:V
            lb = lb - 0.5 * (sum(sum(T{o}.mu .* T{o}.mu)) + N(o) * K) + sum(bW.mu(1, :) * T{o}.mu) + sum(sum((bW.mu(2:R + 1, :)' * Z{o}.mu) .* T{o}.mu)) - 0.5 * (N(o) * trace(WWT.mu * Z{o}.sigma) + sum(sum(Z{o}.mu .* (WWT.mu * Z{o}.mu)))) - 0.5 * N(o) * (bW.mu(1, :) * bW.mu(1, :)' + sum(bW.sigma(1, 1, :))) - sum(Z{o}.mu' * (bW.mu(2:R + 1, :) * bW.mu(1, :)' + sum(bW.sigma(2:R + 1, 1, :), 3))) - 0.5 * N(o) * K * log2pi;
        end

        %%%% q(Phi)
        for o = 1:V
            lb = lb + sum(sum(Phi{o}.alpha + log(Phi{o}.beta) + gammaln(Phi{o}.alpha) + (1 - Phi{o}.alpha) .* psi(Phi{o}.alpha)));
        end
        %%%% q(Q)
        for o = 1:V
            for s = 1:R
                lb = lb + 0.5 * (D(o) * (log2pi + 1) + logdet(Q{o}.sigma(:, :, s)));
            end
        end
        %%%% q(Z)
        for o = 1:V
            lb = lb + 0.5 * N(o) * (R * (log2pi + 1) + logdet(Z{o}.sigma));
        end
        %%%% q(lambda)
        lb = lb + sum(lambda.alpha + log(lambda.beta) + gammaln(lambda.alpha) + (1 - lambda.alpha) .* psi(lambda.alpha));
        %%%% q(Psi)
        lb = lb + sum(sum(Psi.alpha + log(Psi.beta) + gammaln(Psi.alpha) + (1 - Psi.alpha) .* psi(Psi.alpha)));
        %%%% q(b, W)
        for c = 1:K
            lb = lb + 0.5 * ((R + 1) * (log2pi + 1) + logdet(bW.sigma(:, :, c))); 
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