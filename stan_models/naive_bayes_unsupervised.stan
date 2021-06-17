// This has been taken from:
// https://mc-stan.org/docs/2_22/stan-users-guide/naive-bayes-classification-and-clustering.html

data {
  // training data
  int<lower=1> K;               // num topics
  int<lower=1> V;               // num words
  int<lower=0> M;               // num docs
  int<lower=0> N;               // total word instances
  // int<lower=1,upper=K> z[M];    // topic for doc m
  int<lower=1,upper=V> w[N];    // word n
  int<lower=1,upper=M> doc[N];  // doc ID for word n
  // hyperparameters
  vector<lower=0>[K] alpha;     // topic prior
  vector<lower=0>[V] beta;      // word prior
}
parameters {
  positive_ordered[K] lambda; // we need to impose identifiability
  simplex[V] phi[K];            // word dist for topic k
}

transformed parameters {
  simplex[K] theta = lambda / sum(lambda);// topic prevalence
}

model {
  real gamma[M, K]; // If we want gamma, we can move it up to the parameter block
  // theta ~ dirichlet(alpha);
  lambda ~ gamma(alpha, 1);
  for (k in 1:K)
    phi[k] ~ dirichlet(beta);
  for (m in 1:M)
    for (k in 1:K)
      gamma[m, k] = categorical_lpmf(k | theta);
  for (n in 1:N)
    for (k in 1:K)
      gamma[doc[n], k] = gamma[doc[n], k] + categorical_lpmf(w[n] | phi[k]);
  for (m in 1:M)
    target += log_sum_exp(gamma[m]);
}
