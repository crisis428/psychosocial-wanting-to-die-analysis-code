# Run the finalized network-analysis workflow in sequential source files.
# The split is organizational only; all steps share the global environment.
args_all <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args_all, value = TRUE)
SCRIPT_DIR <- if (length(file_arg)) dirname(normalizePath(sub("^--file=", "", file_arg[1]))) else file.path(getwd(), "analysis", "04_network")
STEP_DIR <- file.path(SCRIPT_DIR, "steps")
step_files <- sort(list.files(STEP_DIR, pattern = "^step_[0-9]+\\.R$", full.names = TRUE))
if (length(step_files) == 0L) stop("No network step files found in: ", STEP_DIR)
for (step_file in step_files) {
  cat("\n=== Running ", basename(step_file), " ===\n", sep = "")
  sys.source(step_file, envir = .GlobalEnv)
}
