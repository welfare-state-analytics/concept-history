library(rstan)

# Lets play with a vocabulary of size 26
vocabulary <- letters[1:24]
V <- length(vocabulary)
set.seed(4711)

# Theta 
theta <- c(0.1, 0.2, 0.3, 0.4)
K <- length(theta)

# Phi
# Compute cluster probabilities (using a Dirichlet(0.3))
p <- matrix(rgamma(K*V, shape = 0.3, 1), nrow = 4)
# Manual, simple, structure
p <- matrix(0.1, nrow = K, ncol = V)
p[1,1:6] <- 1;p[2,7:12] <- 1; p[3,13:18] <- 1; p[4,19:24] <- 1

# True parameters used
phi <- p/rowSums(p)


M <- 100 # No of contexts
Nd <- 20 # Tokens in each context

# Simulate clusters
z <- sample(1L:K, size = M, replace = TRUE, prob = theta)

corpus <- list()
for(i in 1:M){
  doc <- rep(i, Nd)
  w <- sample(vocabulary, size = Nd, replace = TRUE, prob = phi[z[i],])
  corpus[[i]] <- list(doc = doc, w = w)
}

doc <- unlist(lapply(corpus, function(x) x$doc))
w <- as.factor(unlist(lapply(corpus, function(x) x$w)))

stan_data <- list(K = K, V = V, M = M, N = length(w), 
                  alpha = rep(0.5, K), beta = rep(0.5, V),
                  z = z, w = as.integer(w), doc = doc)
jsonlite::write_json(stan_data, path = "tests/test_data/test_stan_data.json", auto_unbox = TRUE, pretty = TRUE)
jsonlite::write_json(phi, path = "tests/test_data/test_phi.json", auto_unbox = TRUE, pretty = TRUE)
jsonlite::write_json(phi, path = "tests/test_data/test_theta.json", auto_unbox = TRUE, pretty = TRUE)


# Fit an supervised model
model_fit_nb <- stan(file = "stan_models/naive_bayes.stan", data = stan_data)

# Fit an unsupervised model
model_fit_unb <- stan(file = "stan_models/naive_bayes_unsupervised.stan", data = stan_data)




# Potential improvements:
# 1) Sum up sufficient statistics instead of iterating over words
# 2) Trim vocabulary to simplify


