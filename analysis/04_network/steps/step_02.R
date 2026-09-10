# ----------------------------------------------------------------------------
# 3. General helper functions
# ----------------------------------------------------------------------------

write_csv_utf8 <- function(x, path, row.names = FALSE) {
  write.csv(x, path, row.names = row.names, fileEncoding = "UTF-8")
}

read_checkpoint <- function(path) {
  if (!file.exists(path) || FORCE_RERUN) return(NULL)
  object <- tryCatch(readRDS(path), error = function(e) { message("[checkpoint] Unreadable checkpoint ignored: ", basename(path)); NULL })
  if (is.null(object)) return(NULL)
  if (!is.list(object) || is.null(object$checkpoint_tag) || !identical(object$checkpoint_tag, CHECKPOINT_TAG)) {
    message("[checkpoint] Stale or untagged checkpoint ignored: ", basename(path)); return(NULL)
  }
  object$value
}

save_checkpoint <- function(value, path) {
  temporary_path <- paste0(path, ".tmp-", Sys.getpid())
  on.exit(unlink(temporary_path), add = TRUE)
  saveRDS(list(checkpoint_tag = CHECKPOINT_TAG, created_utc = format(Sys.time(), tz = "UTC", usetz = TRUE), value = value), temporary_path)
  if (file.exists(path) && !file.remove(path)) stop("Could not replace checkpoint: ", path)
  if (!file.rename(temporary_path, path)) stop("Could not publish checkpoint: ", path)
}

load_or_run <- function(path, expression, label) {
  cached <- read_checkpoint(path)
  if (!is.null(cached)) { message("[checkpoint] Loading: ", label); return(cached) }
  message("[run] ", label); value <- force(expression); save_checkpoint(value, path); value
}

run_and_capture_warnings <- function(expression) {
  warnings_seen <- character(0)
  value <- withCallingHandlers(force(expression), warning = function(w) { warnings_seen <<- c(warnings_seen, conditionMessage(w)); invokeRestart("muffleWarning") })
  list(value = value, warnings = unique(warnings_seen))
}

to_numeric_strict <- function(x, variable) {
  if (is.factor(x)) x <- as.character(x)
  result <- suppressWarnings(as.numeric(x))
  bad <- is.na(result) & !is.na(x) & trimws(as.character(x)) != ""
  if (any(bad)) stop(variable, " contains values that cannot be converted to numeric.")
  result
}

rank_desc <- function(x) rank(-x, ties.method = "min", na.last = "keep")

z_score_safe <- function(x) {
  s <- stats::sd(x, na.rm = TRUE)
  if (!is.finite(s) || s == 0) return(rep(0, length(x)))
  (x - mean(x, na.rm = TRUE)) / s
}

empty_edge_table <- function() data.frame(node_1 = character(), node_2 = character(), weight = numeric(), absolute_weight = numeric(), direction = character())

matrix_to_edge_table <- function(W, nonzero_only = TRUE) {
  keep <- upper.tri(W)
  if (nonzero_only) keep <- keep & W != 0
  ind <- which(keep, arr.ind = TRUE)
  if (nrow(ind) == 0L) return(empty_edge_table())
  weights <- W[ind]
  out <- data.frame(node_1 = rownames(W)[ind[, 1]], node_2 = colnames(W)[ind[, 2]], weight = as.numeric(weights), absolute_weight = abs(as.numeric(weights)), direction = ifelse(weights > 0, "positive", ifelse(weights < 0, "negative", "zero")))
  out[order(out$absolute_weight, decreasing = TRUE), , drop = FALSE]
}

make_mixed_cor <- function(dat) {
  C <- qgraph::cor_auto(dat, detectOrdinal = TRUE, ordinalLevelMax = 7, forcePD = TRUE, missing = "listwise", verbose = FALSE)
  dimnames(C) <- list(names(dat), names(dat)); C
}

estimate_ebic_network <- function(dat) {
  bootnet::estimateNetwork(dat, default = "EBICglasso", tuning = EBIC_GAMMA, corMethod = "cor_auto", missing = "listwise", corArgs = list(detectOrdinal = TRUE, ordinalLevelMax = 7, forcePD = TRUE), refit = FALSE, lambda.min.ratio = LAMBDA_MIN_RATIO, nlambda = N_LAMBDA, threshold = FALSE, verbose = FALSE)
}

get_weight_matrix <- function(estimate_object, node_names) {
  W <- as.matrix(estimate_object$graph); diag(W) <- 0; dimnames(W) <- list(node_names, node_names); W
}

# ----------------------------------------------------------------------------
# 4. Layered spring layout
# ----------------------------------------------------------------------------
rescale_to_range <- function(x, lower, upper) {
  observed_range <- range(x, finite = TRUE)
  if (!all(is.finite(observed_range)) || diff(observed_range) == 0) return(rep(mean(c(lower, upper)), length(x)))
  lower + (x - observed_range[1]) / diff(observed_range) * (upper - lower)
}

make_separate_spring_layout <- function(W, seed) {
  node_names <- colnames(W); predictor_names <- setdiff(node_names, OUTCOME)
  predictor_matrix <- W[predictor_names, predictor_names, drop = FALSE]
  set.seed(seed)
  predictor_qgraph <- qgraph::qgraph(predictor_matrix, layout = "spring", DoNotPlot = TRUE, labels = predictor_names, groups = NULL, minimum = 0, cut = 0, threshold = 0, details = FALSE)
  raw_predictor_layout <- if (!is.null(predictor_qgraph$layout.orig)) as.matrix(predictor_qgraph$layout.orig) else { warning("qgraph did not return layout.orig; using qobj$layout as fallback."); as.matrix(predictor_qgraph$layout) }
  rownames(raw_predictor_layout) <- predictor_names; colnames(raw_predictor_layout) <- c("x", "y")
  transformed_layout <- sweep(raw_predictor_layout, 2, colMeans(raw_predictor_layout), "-")
  if (nrow(transformed_layout) >= 3L && all(is.finite(transformed_layout))) {
    principal_components <- stats::prcomp(transformed_layout, center = FALSE, scale. = FALSE)
    transformed_layout <- principal_components$x[, 1:2, drop = FALSE]; rownames(transformed_layout) <- predictor_names
  }
  psychological_nodes <- predictor_names[NODE_DOMAIN[predictor_names] == "Psychological"]
  demographic_nodes <- predictor_names[NODE_DOMAIN[predictor_names] == "Demographic"]
  if (length(psychological_nodes) > 0L && length(demographic_nodes) > 0L) {
    psychological_mean_x <- mean(transformed_layout[psychological_nodes, 1]); demographic_mean_x <- mean(transformed_layout[demographic_nodes, 1])
    if (psychological_mean_x > demographic_mean_x) transformed_layout[, 1] <- -transformed_layout[, 1]
  }
  transformed_layout[, 1] <- rescale_to_range(transformed_layout[, 1], lower = -1.32, upper = 1.32)
  transformed_layout[, 2] <- rescale_to_range(transformed_layout[, 2], lower = -0.56, upper = -0.12)
  final_layout <- matrix(NA_real_, nrow = length(node_names), ncol = 2, dimnames = list(node_names, c("x", "y")))
  final_layout[OUTCOME, ] <- c(0, 1.05); final_layout[predictor_names, ] <- transformed_layout[predictor_names, , drop = FALSE]
  raw_distances <- as.numeric(stats::dist(raw_predictor_layout)); layered_distances <- as.numeric(stats::dist(transformed_layout))
  distance_correlation <- if (length(raw_distances) > 1L && stats::sd(raw_distances) > 0 && stats::sd(layered_distances) > 0) stats::cor(raw_distances, layered_distances) else NA_real_
  attr(final_layout, "layout_method") <- paste0("predictor-only spring; global PCA orientation/reflection; ", "intentional vertical compression; outcome fixed above predictor layer")
  attr(final_layout, "si_vertical_clearance") <- min(final_layout[OUTCOME, 2] - final_layout[predictor_names, 2])
  attr(final_layout, "predictor_horizontal_span") <- diff(range(final_layout[predictor_names, 1]))
  attr(final_layout, "predictor_vertical_span") <- diff(range(final_layout[predictor_names, 2]))
  attr(final_layout, "predictor_distance_correlation") <- distance_correlation
  list(raw = raw_predictor_layout, oriented = final_layout)
}

# ----------------------------------------------------------------------------
# 5. Centrality functions
# ----------------------------------------------------------------------------
predictor_only_bridge_ei <- function(W) {
  predictor_names <- setdiff(colnames(W), OUTCOME); P <- W[predictor_names, predictor_names, drop = FALSE]
  communities <- unname(NODE_DOMAIN[predictor_names]); names(communities) <- predictor_names
  bridge_ei <- vapply(seq_along(predictor_names), function(i) sum(P[i, communities != communities[i]], na.rm = TRUE), numeric(1))
  names(bridge_ei) <- predictor_names; bridge_ei
}

make_centrality_table <- function(W, group_name) {
  strength <- rowSums(abs(W)); expected_influence <- rowSums(W); bridge_ei <- predictor_only_bridge_ei(W)
  out <- data.frame(group = group_name, node = colnames(W), node_code = unname(NODE_CODE[colnames(W)]), node_name = unname(NODE_NAME[colnames(W)]), domain = unname(NODE_DOMAIN[colnames(W)]), strength = as.numeric(strength[colnames(W)]), expected_influence_1step = as.numeric(expected_influence[colnames(W)]), bridge_expected_influence_1step_predictor_only = NA_real_)
  out$bridge_expected_influence_1step_predictor_only[match(names(bridge_ei), out$node)] <- as.numeric(bridge_ei)
  out$strength_rank <- rank_desc(out$strength); out$expected_influence_rank <- rank_desc(out$expected_influence_1step); out$bridge_expected_influence_rank <- rank_desc(out$bridge_expected_influence_1step_predictor_only)
  out
}
