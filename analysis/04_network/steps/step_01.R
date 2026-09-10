# NOTE FOR THIS CODE COLLECTION
# - Numeric network estimates, centrality, bridge EI, and bootstrap outputs are
#   produced by this script.
# - The final manuscript Figure 3 is generated separately with
#   ../../reporting/figures/figure3_network_fixed_layout.py.
# - That reporting script uses the total-sample coordinates for all three panels;
#   layout coordinates generated here are analysis/diagnostic outputs only.
# - Edit WORK_DIR before execution. For a short code check, set TEST_MODE <- TRUE.

# ============================================================================
# Network analysis: final reproducible pipeline
# ============================================================================
# Primary analysis locked in this script:
#   - source data: primary_cc.csv (N = 6,605 expected)
#   - total, male, and female networks
#   - mixed correlation matrix + EBICglasso (gamma = 0.50)
#   - no post-estimation P-value or magnitude filtering
#   - predictor-only spring layouts transformed into a lower visual layer,
#     with the focal outcome displayed separately at the top
#   - red positive edges, blue negative edges, common edge-width scale
#   - strength, 1-step expected influence, predictor-only bridge EI
#   - nonparametric edge bootstrap and case-dropping centrality bootstrap
#
# IMPORTANT
# Full bootstrap execution is computationally expensive. First run TEST_MODE =
# TRUE. When that completes, change TEST_MODE to FALSE and run the full analysis.
# Checkpoints prevent completed stages from being repeated.
# ============================================================================


# ----------------------------------------------------------------------------
# 0. User settings
# ----------------------------------------------------------------------------

WORK_DIR <- getwd()  # set this explicitly if running from another directory
DATA_FILE <- "primary_cc.csv"

TEST_MODE <- FALSE         # TRUE: code check; FALSE: final analysis
FORCE_RERUN <- FALSE       # TRUE: ignore checkpoints and rerun everything
INSTALL_MISSING <- TRUE
CHECKPOINT_TAG_BASE <- "network_final_20260806_v1"

SEED <- 20260806L
EBIC_GAMMA <- 0.50
N_LAMBDA <- 100L
LAMBDA_MIN_RATIO <- 0.01

N_BOOT_EDGE_FINAL <- 2000L
N_BOOT_CASE_FINAL <- 1000L
N_BOOT_EDGE_TEST <- 20L
N_BOOT_CASE_TEST <- 20L

CASE_MIN <- 0.05
CASE_MAX <- 0.75
CASE_LEVELS <- 10L

detected_cores <- parallel::detectCores(logical = TRUE)
if (!is.finite(detected_cores)) detected_cores <- 2L
N_CORES <- max(1L, min(4L, as.integer(detected_cores) - 1L))

if (TEST_MODE) {
  N_BOOT_EDGE <- N_BOOT_EDGE_TEST
  N_BOOT_CASE <- N_BOOT_CASE_TEST
  OUTPUT_FOLDER <- "network_results_TEST"
} else {
  N_BOOT_EDGE <- N_BOOT_EDGE_FINAL
  N_BOOT_CASE <- N_BOOT_CASE_FINAL
  OUTPUT_FOLDER <- "network_results_FINAL_v1"
}

CHECKPOINT_TAG <- paste(
  CHECKPOINT_TAG_BASE,
  ifelse(TEST_MODE, "TEST", "FINAL"),
  paste0("gamma", EBIC_GAMMA),
  paste0("nlambda", N_LAMBDA),
  paste0("lmr", LAMBDA_MIN_RATIO),
  paste0("edge", N_BOOT_EDGE),
  paste0("case", N_BOOT_CASE),
  paste0("seed", SEED),
  sep = "_"
)


# ----------------------------------------------------------------------------
# 1. Working directory and packages
# ----------------------------------------------------------------------------

if (!dir.exists(WORK_DIR)) {
  stop(
    "Working directory not found:\n", WORK_DIR,
    "\nCheck the spelling before running the analysis."
  )
}
setwd(WORK_DIR)

required_packages <- c(
  "qgraph", "bootnet", "networktools", "ggplot2"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0L) {
  if (!INSTALL_MISSING) {
    stop("Install these packages first: ", paste(missing_packages, collapse = ", "))
  }
  install.packages(missing_packages, dependencies = TRUE)
}

suppressPackageStartupMessages({
  library(qgraph)
  library(bootnet)
  library(networktools)
  library(ggplot2)
})

set.seed(SEED)

output_dir <- file.path(getwd(), OUTPUT_FOLDER)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

capture.output(sessionInfo(), file = file.path(output_dir, "00_session_info.txt"))


# ----------------------------------------------------------------------------
# 2. Constants: variables, labels, domains, and colors
# ----------------------------------------------------------------------------

OUTCOME <- "py_thoughts_wanting_to_die"
SEX_VAR <- "sex_female"

NETWORK_VARS_TOTAL <- c(
  OUTCOME,
  "gad7_total",
  "pss14_total",
  "low_self_esteem_score",
  "poor_subjective_physical_health",
  "poor_subjective_mental_health",
  "low_sense_of_belonging",
  "low_perceived_social_equality",
  "low_social_trust",
  "age_years",
  SEX_VAR,
  "married_current",
  "employed_corrected",
  "living_alone"
)

NODE_CODE <- c(
  py_thoughts_wanting_to_die = "SI",
  gad7_total = "GA",
  pss14_total = "PS",
  low_self_esteem_score = "SE",
  poor_subjective_physical_health = "PH",
  poor_subjective_mental_health = "MH",
  low_sense_of_belonging = "BE",
  low_perceived_social_equality = "EQ",
  low_social_trust = "TR",
  age_years = "AG",
  sex_female = "FE",
  married_current = "MA",
  employed_corrected = "EM",
  living_alone = "LA"
)

NODE_NAME <- c(
  py_thoughts_wanting_to_die = "Past-year thoughts of wanting to die",
  gad7_total = "GAD-7 score",
  pss14_total = "PSS-14 score",
  low_self_esteem_score = "Low self-esteem score",
  poor_subjective_physical_health = "Poor subjective physical health",
  poor_subjective_mental_health = "Poor subjective mental health",
  low_sense_of_belonging = "Low sense of belonging",
  low_perceived_social_equality = "Low perceived social equality",
  low_social_trust = "Low social trust",
  age_years = "Age",
  sex_female = "Female sex",
  married_current = "Currently married",
  employed_corrected = "Currently working",
  living_alone = "Living alone"
)

NODE_DOMAIN <- c(
  py_thoughts_wanting_to_die = "Outcome",
  gad7_total = "Psychological",
  pss14_total = "Psychological",
  low_self_esteem_score = "Psychological",
  poor_subjective_physical_health = "Subjective health",
  poor_subjective_mental_health = "Subjective health",
  low_sense_of_belonging = "Social-contextual",
  low_perceived_social_equality = "Social-contextual",
  low_social_trust = "Social-contextual",
  age_years = "Demographic",
  sex_female = "Demographic",
  married_current = "Demographic",
  employed_corrected = "Demographic",
  living_alone = "Demographic"
)

DOMAIN_COLOR <- c(
  Outcome = "#000000",
  Psychological = "#A65628",
  `Subjective health` = "#1F9E89",
  `Social-contextual` = "#4DAF4A",
  Demographic = "#8E44AD"
)

POSITIVE_EDGE_COLOR <- "#D73027"  # red = positive
NEGATIVE_EDGE_COLOR <- "#4575B4"  # blue = negative

group_specs <- list(
  total = list(label = "Total sample", sex = NA_integer_, seed = SEED + 10L),
  male = list(label = "Male subgroup", sex = 0L, seed = SEED + 20L),
  female = list(label = "Female subgroup", sex = 1L, seed = SEED + 30L)
)

