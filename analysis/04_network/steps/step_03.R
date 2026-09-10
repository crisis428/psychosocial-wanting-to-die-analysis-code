# ----------------------------------------------------------------------------
# 6. Read and validate source data
# ----------------------------------------------------------------------------

if (!file.exists(DATA_FILE)) {
  stop("Exact input file not found: ", DATA_FILE, "\nPlace it in:\n", normalizePath(getwd(), winslash = "/", mustWork = FALSE))
}

data_path <- DATA_FILE
source_md5 <- unname(tools::md5sum(data_path))
CHECKPOINT_TAG <- paste0(CHECKPOINT_TAG, "_md5", substr(source_md5, 1L, 12L))
raw_data <- tryCatch(read.csv(data_path, stringsAsFactors = FALSE, check.names = FALSE, fileEncoding = "UTF-8-BOM"), error = function(e) read.csv(data_path, stringsAsFactors = FALSE, check.names = FALSE))
missing_variables <- setdiff(c("participant_id", NETWORK_VARS_TOTAL), names(raw_data))
if (length(missing_variables) > 0L) stop("Required variables missing: ", paste(missing_variables, collapse = ", "))
participant_id <- trimws(as.character(raw_data$participant_id))
if (any(is.na(participant_id) | participant_id == "")) stop("participant_id contains missing or blank values.")
if (anyDuplicated(participant_id) > 0L) stop("participant_id must be unique; duplicates were detected.")
analysis_data <- raw_data[, NETWORK_VARS_TOTAL, drop = FALSE]
for (v in names(analysis_data)) analysis_data[[v]] <- to_numeric_strict(analysis_data[[v]], v)
n_source <- nrow(analysis_data); complete_rows <- complete.cases(analysis_data); analysis_data <- analysis_data[complete_rows, , drop = FALSE]; n_complete <- nrow(analysis_data)
binary_vars <- c(OUTCOME, SEX_VAR, "married_current", "employed_corrected", "living_alone")
for (v in binary_vars) {
  observed <- sort(unique(analysis_data[[v]]))
  if (!all(observed %in% c(0, 1))) stop(v, " must be coded 0/1. Observed: ", paste(observed, collapse = ", "))
}
zero_variance <- names(analysis_data)[vapply(analysis_data, function(x) length(unique(x)) < 2L, logical(1))]
if (length(zero_variance) > 0L) stop("Zero-variance variables: ", paste(zero_variance, collapse = ", "))
if (n_source != 6605L || n_complete != 6605L) stop("Expected exactly 6,605 complete records. Source N = ", n_source, "; complete-case N = ", n_complete, ".")
expected_ranges <- data.frame(variable = NETWORK_VARS_TOTAL, expected_min = c(0,0,0,10,1,1,1,1,1,20,0,0,0,0), expected_max = c(1,21,56,40,4,4,4,4,4,69,1,1,1,1))
for (i in seq_len(nrow(expected_ranges))) {
  v <- expected_ranges$variable[i]; observed_min <- min(analysis_data[[v]]); observed_max <- max(analysis_data[[v]])
  if (observed_min != expected_ranges$expected_min[i] || observed_max != expected_ranges$expected_max[i]) stop("Range check failed for ", v, ": observed ", observed_min, " to ", observed_max, "; expected ", expected_ranges$expected_min[i], " to ", expected_ranges$expected_max[i], ".")
}
male_n <- sum(analysis_data[[SEX_VAR]] == 0); female_n <- sum(analysis_data[[SEX_VAR]] == 1)
if (male_n != 3181L || female_n != 3424L) stop("Sex subgroup count check failed: male = ", male_n, "; female = ", female_n, ". Expected 3,181 and 3,424.")
qc <- data.frame(source_file = data_path, source_md5 = source_md5, source_rows = n_source, complete_case_rows = n_complete, excluded_in_script = n_source - n_complete, outcome_positive_n = sum(analysis_data[[OUTCOME]] == 1), outcome_positive_percent = 100 * mean(analysis_data[[OUTCOME]] == 1), participant_id_unique = !anyDuplicated(participant_id), male_n = male_n, female_n = female_n)
write_csv_utf8(qc, file.path(output_dir, "01_data_qc_summary.csv"))
write_csv_utf8(expected_ranges, file.path(output_dir, "01_expected_variable_ranges.csv"))
variable_qc <- data.frame(variable = names(analysis_data), n = vapply(analysis_data, length, integer(1)), unique_values = vapply(analysis_data, function(x) length(unique(x)), integer(1)), mean = vapply(analysis_data, mean, numeric(1)), sd = vapply(analysis_data, stats::sd, numeric(1)), minimum = vapply(analysis_data, min, numeric(1)), maximum = vapply(analysis_data, max, numeric(1)))
write_csv_utf8(variable_qc, file.path(output_dir, "02_variable_qc.csv"))
node_dictionary <- data.frame(variable = names(NODE_CODE), code = unname(NODE_CODE), label = unname(NODE_NAME), domain = unname(NODE_DOMAIN))
write_csv_utf8(node_dictionary, file.path(output_dir, "03_node_dictionary.csv"))
