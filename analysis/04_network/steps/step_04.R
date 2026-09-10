# ----------------------------------------------------------------------------
# 7. Estimate total, male, and female networks
# ----------------------------------------------------------------------------
network_results <- list()
for (group_name in names(group_specs)) {
  spec <- group_specs[[group_name]]
  group_dir <- file.path(output_dir, group_name)
  dir.create(group_dir, recursive = TRUE, showWarnings = FALSE)
  if (is.na(spec$sex)) {
    dat <- analysis_data; vars <- NETWORK_VARS_TOTAL
  } else {
    dat <- analysis_data[analysis_data[[SEX_VAR]] == spec$sex, , drop = FALSE]
    vars <- setdiff(NETWORK_VARS_TOTAL, SEX_VAR)
    dat <- dat[, vars, drop = FALSE]
  }
  dat <- dat[, vars, drop = FALSE]
  if (any(vapply(dat, function(x) length(unique(x)) < 2L, logical(1)))) stop("A zero-variance node was found in the ", group_name, " network.")
  mixed_cor <- load_or_run(file.path(group_dir, "10_mixed_correlation_matrix.rds"), make_mixed_cor(dat), paste0(group_name, ": mixed correlation matrix"))
  estimate_object <- load_or_run(file.path(group_dir, "11_network_estimate_bootnet_object.rds"), estimate_ebic_network(dat), paste0(group_name, ": EBICglasso network"))
  W <- get_weight_matrix(estimate_object, vars)
  direct_qgraph_W <- qgraph::EBICglasso(S = mixed_cor, n = nrow(dat), gamma = EBIC_GAMMA, nlambda = N_LAMBDA, lambda.min.ratio = LAMBDA_MIN_RATIO, threshold = FALSE, refit = FALSE, verbose = FALSE)
  diag(direct_qgraph_W) <- 0; dimnames(direct_qgraph_W) <- list(vars, vars)
  crosscheck_max_difference <- max(abs(W - direct_qgraph_W))
  if (crosscheck_max_difference > 1e-8) stop("bootnet and direct qgraph EBICglasso matrices differ in the ", group_name, " network (maximum absolute difference = ", crosscheck_max_difference, ").")
  layout_object <- load_or_run(file.path(group_dir, "12_predictor_spring_and_final_layered_layout.rds"), make_separate_spring_layout(W, spec$seed), paste0(group_name, ": predictor spring and final layered layout"))
  raw_layout <- layout_object$raw; layout <- layout_object$oriented
  centrality <- make_centrality_table(W, group_name)
  all_edges <- matrix_to_edge_table(W, nonzero_only = FALSE)
  nonzero_edges <- matrix_to_edge_table(W, nonzero_only = TRUE)
  outcome_edges <- all_edges[all_edges$node_1 == OUTCOME | all_edges$node_2 == OUTCOME, , drop = FALSE]
  outcome_edges$predictor <- ifelse(outcome_edges$node_1 == OUTCOME, outcome_edges$node_2, outcome_edges$node_1)
  outcome_edges <- outcome_edges[order(outcome_edges$absolute_weight, decreasing = TRUE), , drop = FALSE]
  write_csv_utf8(mixed_cor, file.path(group_dir, "10_mixed_correlation_matrix.csv"), TRUE)
  write_csv_utf8(W, file.path(group_dir, "11_EBICglasso_weight_matrix.csv"), TRUE)
  write_csv_utf8(direct_qgraph_W, file.path(group_dir, "11_direct_qgraph_EBICglasso_crosscheck_matrix.csv"), TRUE)
  write_csv_utf8(data.frame(node = rownames(raw_layout), x = raw_layout[,1], y = raw_layout[,2]), file.path(group_dir, "12a_predictor_spring_raw_coordinates.csv"))
  write_csv_utf8(data.frame(node = rownames(layout), x = layout[,1], y = layout[,2]), file.path(group_dir, "12b_final_layered_layout_coordinates.csv"))
  write_csv_utf8(all_edges, file.path(group_dir, "13_all_possible_edges.csv"))
  write_csv_utf8(nonzero_edges, file.path(group_dir, "14_nonzero_edges.csv"))
  write_csv_utf8(outcome_edges, file.path(group_dir, "15_SI_incident_edges.csv"))
  write_csv_utf8(centrality, file.path(group_dir, "16_centrality_raw_and_ranks.csv"))
  network_results[[group_name]] <- list(name = group_name, label = spec$label, data = dat, n = nrow(dat), variables = vars, correlation = mixed_cor, estimate = estimate_object, weights = W, direct_qgraph_weights = direct_qgraph_W, crosscheck_max_difference = crosscheck_max_difference, raw_layout = raw_layout, layout = layout, centrality = centrality, nonzero_edges = nonzero_edges, outcome_edges = outcome_edges, group_dir = group_dir, seed = spec$seed)
}
sample_sizes <- do.call(rbind, lapply(network_results, function(x) {
  upper_weights <- x$weights[upper.tri(x$weights)]
  eigenvalues <- eigen(x$correlation, symmetric = TRUE, only.values = TRUE)$values
  data.frame(group = x$name, label = x$label, n = x$n, nodes = ncol(x$data), nonzero_edges = nrow(x$nonzero_edges), positive_nonzero_edges = sum(upper_weights > 0), negative_nonzero_edges = sum(upper_weights < 0), correlation_minimum_eigenvalue = min(eigenvalues), correlation_condition_number = max(eigenvalues)/min(eigenvalues), correlation_maximum_symmetry_error = max(abs(x$correlation-t(x$correlation))), ebic_matrix_crosscheck_max_difference = x$crosscheck_max_difference, layout_method = attr(x$layout,"layout_method"), si_vertical_clearance = attr(x$layout,"si_vertical_clearance"), si_strictly_top = attr(x$layout,"si_vertical_clearance") >= -1e-6, predictor_horizontal_span = attr(x$layout,"predictor_horizontal_span"), predictor_vertical_span = attr(x$layout,"predictor_vertical_span"), predictor_distance_correlation = attr(x$layout,"predictor_distance_correlation"))
}))
write_csv_utf8(sample_sizes, file.path(output_dir, "04_network_sample_sizes.csv"))
