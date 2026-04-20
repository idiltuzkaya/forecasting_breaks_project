# ============================================================
# ARIMA Simulation Study
# DGP: AR(1) with mean shift (level shift only)
# Break sizes: small (10%), medium (25%), large (50%)
# n = 200, 500 replications, break at t = 100
# Strategies: full-sample, rolling (26/52/104), recursive
# Metrics: RMSE, MAE
# ============================================================

library(forecast)
library(parallel)
library(doParallel)
library(foreach)

# ============================================================
# PART 1: DATA GENERATION
# ============================================================

set.seed(42)
start <- Sys.time()
n         <- 200
n_reps    <- 500
break_at  <- 100
phi       <- 0.5
sigma     <- 1
mu1       <- 0

break_sizes <- list(
  small  = 1.0,   # |mu2 - mu1| / sigma = 1
  medium = 2.5,   # |mu2 - mu1| / sigma = 2.5
  large  = 5.0    # |mu2 - mu1| / sigma = 5
)

generate_series <- function(n, break_at, phi, sigma, mu1, mu2) {
  y <- numeric(n)
  y[1] <- mu1 + rnorm(1, 0, sigma)
  for (t in 2:n) {
    mu_t  <- ifelse(t  < break_at, mu1, mu2)
    mu_tm1 <- ifelse(t-1 < break_at, mu1, mu2)
    y[t]  <- mu_t + phi * (y[t-1] - mu_tm1) + rnorm(1, 0, sigma)
  }
  return(y)
}

sim_data <- list()

for (size_name in names(break_sizes)) {
  mu2 <- mu1 + break_sizes[[size_name]] * sigma
  replications <- matrix(NA, nrow = n, ncol = n_reps)
  for (r in 1:n_reps) {
    replications[, r] <- generate_series(n, break_at, phi, sigma, mu1, mu2)
  }
  sim_data[[size_name]] <- replications
}

cat("Data generation complete.\n")
cat("sim_data structure: 3 break sizes, each a", nrow(sim_data[[1]]), "x", ncol(sim_data[[1]]), "matrix\n")

# ============================================================
# PART 2: FORECASTING
# ============================================================

n_cores <- detectCores() - 1
cl <- makeCluster(n_cores)
registerDoParallel(cl)
cat("Running on", n_cores, "cores\n")

train_end <- 140       # 70% train, 30% holdout (60 periods)
h_vec     <- c(1, 4, 8)

# ── Core forecasting function ─────────────────────────────────
forecast_arima <- function(y, strategy, window_size = 52, h = 1) {
  n       <- length(y)
  origins <- train_end:(n - h)
  preds   <- numeric(length(origins))
  
  for (i in seq_along(origins)) {
    origin <- origins[i]
    train  <- switch(strategy,
                     full      = y[1:origin],
                     rolling   = y[max(1, origin - window_size + 1):origin],
                     recursive = y[1:origin]
    )
    fit <- tryCatch(
      forecast::auto.arima(train, stepwise = TRUE, approximation = TRUE),
      error = function(e) NULL
    )
    preds[i] <- if (is.null(fit)) NA else forecast::forecast(fit, h = h)$mean[h]
  }
  
  actual <- y[origins + h]
  rmse   <- sqrt(mean((actual - preds)^2, na.rm = TRUE))
  mae    <- mean(abs(actual - preds),     na.rm = TRUE)
  
  return(c(rmse = rmse, mae = mae))
}

# ── Strategy definitions ──────────────────────────────────────
strategies <- list(
  full        = list(strategy = "full",      window = NA),
  rolling_26  = list(strategy = "rolling",   window = 26),
  rolling_52  = list(strategy = "rolling",   window = 52),
  rolling_104 = list(strategy = "rolling",   window = 104),
  recursive   = list(strategy = "recursive", window = NA)
)

# ── Main loop ─────────────────────────────────────────────────
results <- list()

for (size_name in names(sim_data)) {
  cat("\n>>> Break size:", size_name, "\n")
  results[[size_name]] <- list()
  
  for (strat_name in names(strategies)) {
    cat("  Strategy:", strat_name, "\n")
    s <- strategies[[strat_name]]
    results[[size_name]][[strat_name]] <- list()
    
    for (h in h_vec) {
      res_mat <- foreach(
        r         = 1:n_reps,
        .packages = "forecast",
        .combine  = rbind
      ) %dopar% {
        forecast_arima(
          y           = sim_data[[size_name]][, r],
          strategy    = s$strategy,
          window_size = ifelse(is.na(s$window), 52, s$window),
          h           = h
        )
      }
      rownames(res_mat) <- NULL
      results[[size_name]][[strat_name]][[paste0("h", h)]] <- res_mat
    }
  }
}

stopCluster(cl)
cat("\nClustered stopped.\n")

# ============================================================
# PART 3: SAVE RESULTS
# ============================================================

saveRDS(results, "simulation_results_arima.rds")
cat("Results saved to simulation_results_arima.rds\n")

# ============================================================
# PART 4: SUMMARY TABLE
# ============================================================

cat("\n========== MEAN RMSE & MAE SUMMARY ==========\n")
for (size_name in names(results)) {
  cat("\nBreak size:", size_name, "\n")
  for (strat in names(results[[size_name]])) {
    for (h_name in names(results[[size_name]][[strat]])) {
      mat       <- results[[size_name]][[strat]][[h_name]]
      mean_rmse <- mean(mat[, "rmse"], na.rm = TRUE)
      mean_mae  <- mean(mat[, "mae"],  na.rm = TRUE)
      cat(sprintf("  %-15s %s  →  RMSE = %.4f  |  MAE = %.4f\n",
                  strat, h_name, mean_rmse, mean_mae))
    }
  }
}

end <- Sys.time()


end- start




