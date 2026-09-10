# ============================================================================
# Supplementary network figures (S5-S7): plotting-only script
# ============================================================================
# 이 스크립트는 네트워크 추정이나 bootstrap을 다시 실행하지 않습니다.
# network_results_FINAL_v1에 저장된 최종 CSV/RDS 객체를 불러와
# Figure S5, Figure S6, Figure S7만 다시 생성합니다.
# ============================================================================

# ----------------------------------------------------------------------------
# 0. 사용자 경로
# ----------------------------------------------------------------------------

WORK_DIR <- getwd()  # set this explicitly if running from another directory
RESULT_FOLDER <- "network_results_FINAL_v1"
RESULT_ZIP_NAME <- "network_results_FINAL_v1.zip"
OUTPUT_FOLDER <- "supplementary_network_figures"

if (!dir.exists(WORK_DIR)) {
  stop("작업 폴더를 찾을 수 없습니다:\n", WORK_DIR)
}

setwd(WORK_DIR)

RESULT_DIR <- file.path(WORK_DIR, RESULT_FOLDER)
RESULT_ZIP <- file.path(WORK_DIR, RESULT_ZIP_NAME)

# 결과 폴더가 없고 ZIP만 있으면 자동으로 압축 해제
if (!dir.exists(RESULT_DIR)) {
  if (file.exists(RESULT_ZIP)) {
    dir.create(RESULT_DIR, recursive = TRUE, showWarnings = FALSE)
    unzip(RESULT_ZIP, exdir = RESULT_DIR)
  } else {
    stop(
      "최종 결과 폴더 또는 ZIP을 찾을 수 없습니다.\n",
      "필요한 폴더: ", RESULT_DIR, "\n",
      "또는 ZIP: ", RESULT_ZIP
    )
  }
}

OUTPUT_DIR <- file.path(RESULT_DIR, OUTPUT_FOLDER)
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

# ----------------------------------------------------------------------------
# 1. 패키지
# ----------------------------------------------------------------------------

required_packages <- c("ggplot2", "patchwork", "bootnet")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0L) {
  install.packages(missing_packages, dependencies = TRUE)
}

suppressPackageStartupMessages({
  library(ggplot2)
  library(patchwork)
  library(bootnet)
})

# ----------------------------------------------------------------------------
# 2. 공통 설정과 검증
# ----------------------------------------------------------------------------

GROUP_ORDER <- c("total", "male", "female")
GROUP_TITLE <- c(
  total = "Total sample",
  male = "Male subgroup",
  female = "Female subgroup"
)

OUTCOME <- "py_thoughts_wanting_to_die"
SEX_VAR <- "sex_female"

NODE_LABEL <- c(
  py_thoughts_wanting_to_die = "Thoughts of wanting to die",
  gad7_total = "GAD-7 score",
  pss14_total = "PSS-14 score",
  low_self_esteem_score = "Low self-esteem",
  poor_subjective_physical_health = "Poor subjective physical health",
  poor_subjective_mental_health = "Poor subjective mental health",
  low_sense_of_belonging = "Low sense of belonging",
  low_perceived_social_equality = "Low perceived social equality",
  low_social_trust = "Low social trust",
  age_years = "Age",
  sex_female = "Female sex",
  married_current = "Married",
  employed_corrected = "Employed",
  living_alone = "Living alone"
)

NODE_ORDER_TOP_TO_BOTTOM <- c(
  OUTCOME,
  "pss14_total",
  "poor_subjective_physical_health",
  "poor_subjective_mental_health",
  "married_current",
  "low_social_trust",
  "low_sense_of_belonging",
  "low_self_esteem_score",
  "low_perceived_social_equality",
  "living_alone",
  "gad7_total",
  SEX_VAR,
  "employed_corrected",
  "age_years"
)

centrality_file <- file.path(RESULT_DIR, "20_centrality_all_groups.csv")
if (!file.exists(centrality_file)) stop("중앙성 결과 파일이 없습니다: ", centrality_file)
centrality <- read.csv(centrality_file, stringsAsFactors = FALSE, check.names = FALSE)

required_columns <- c("group","node","strength","expected_influence_1step","bridge_expected_influence_1step_predictor_only")
missing_columns <- setdiff(required_columns, names(centrality))
if (length(missing_columns) > 0L) stop("중앙성 CSV에 필요한 열이 없습니다: ", paste(missing_columns, collapse = ", "))

expected_group_n <- c(total = 14L, male = 13L, female = 13L)
observed_group_n <- table(centrality$group)
for (g in GROUP_ORDER) {
  if (!g %in% names(observed_group_n)) stop("중앙성 결과에 그룹이 없습니다: ", g)
  if (unname(observed_group_n[g]) != expected_group_n[g]) stop(g, " 그룹의 노드 수가 예상과 다릅니다. 관측=", unname(observed_group_n[g]), ", 예상=", expected_group_n[g])
}
if (any(centrality$group %in% c("male", "female") & centrality$node == SEX_VAR)) stop("성별 하위집단 중앙성 결과에 Female sex 노드가 포함되어 있습니다.")
outcome_bridge <- centrality[centrality$node == OUTCOME,"bridge_expected_influence_1step_predictor_only"]
if (any(!is.na(outcome_bridge))) stop("Outcome의 predictor-only bridge EI는 NA여야 합니다.")

z_score_safe <- function(x) {
  s <- stats::sd(x, na.rm = TRUE)
  if (!is.finite(s) || s == 0) return(rep(0, length(x)))
  (x - mean(x, na.rm = TRUE)) / s
}

add_padding <- function(x, proportion = 0.06, minimum = 0.05) {
  r <- range(x, na.rm = TRUE, finite = TRUE)
  if (!all(is.finite(r))) stop("축 범위를 계산할 수 없습니다.")
  span <- diff(r); pad <- max(minimum, span * proportion); r + c(-pad, pad)
}

# Figure S5
make_efigure5_data <- function(group_name) {
  d <- centrality[centrality$group == group_name, , drop = FALSE]
  d_long <- rbind(
    data.frame(group = group_name,node = d$node,metric = "Strength",value = z_score_safe(d$strength),stringsAsFactors = FALSE),
    data.frame(group = group_name,node = d$node,metric = "Expected influence",value = z_score_safe(d$expected_influence_1step),stringsAsFactors = FALSE)
  )
  group_order <- intersect(NODE_ORDER_TOP_TO_BOTTOM, d$node)
  factor_levels <- rev(unname(NODE_LABEL[group_order]))
  d_long$node_label <- unname(NODE_LABEL[d_long$node])
  d_long$node_label <- factor(d_long$node_label, levels = factor_levels)
  d_long$metric <- factor(d_long$metric, levels = c("Strength", "Expected influence"))
  d_long
}

efigure5_data <- do.call(rbind,lapply(GROUP_ORDER, make_efigure5_data))
efigure5_limits <- add_padding(efigure5_data$value, proportion = 0.05, minimum = 0.10)

make_efigure5_panel <- function(group_name, panel_letter) {
  d <- make_efigure5_data(group_name)
  ggplot(d,aes(x = node_label, y = value, group = 1)) +
    geom_hline(yintercept = 0,linewidth = 0.35,color = "grey55") +
    geom_line(linewidth = 0.50,color = "grey25") +
    geom_point(size = 1.55,color = "black") + coord_flip() +
    facet_wrap(~metric,nrow = 1,scales = "fixed") + scale_x_discrete(drop = TRUE) +
    scale_y_continuous(limits = efigure5_limits,expand = expansion(mult = c(0.01, 0.01))) +
    labs(title = paste0(panel_letter, ". ", GROUP_TITLE[group_name]),x = NULL,y = "Standardized centrality (z score)") +
    theme_grey(base_size = 9) +
    theme(plot.title = element_text(face = "bold",size = 10,hjust = 0),strip.text = element_text(face = "bold",size = 9),axis.text.y = element_text(size = 7.4),axis.text.x = element_text(size = 7.2),axis.title.x = element_text(size = 8.5),panel.spacing.x = grid::unit(0.45, "lines"),plot.margin = margin(5.5, 8, 5.5, 8))
}

efigure5 <- make_efigure5_panel("total", "A") / make_efigure5_panel("male", "B") / make_efigure5_panel("female", "C") + plot_layout(heights = c(1,1,1))

# Figure S6
make_efigure6_data <- function(group_name) {
  d <- centrality[centrality$group == group_name & !is.na(centrality$bridge_expected_influence_1step_predictor_only),,drop = FALSE]
  if (any(d$node == OUTCOME)) stop(group_name, " 그룹의 bridge EI 데이터에 outcome이 포함되어 있습니다.")
  group_order <- intersect(setdiff(NODE_ORDER_TOP_TO_BOTTOM, OUTCOME),d$node)
  factor_levels <- rev(unname(NODE_LABEL[group_order]))
  out <- data.frame(group = group_name,node = d$node,node_label = unname(NODE_LABEL[d$node]),value = d$bridge_expected_influence_1step_predictor_only,stringsAsFactors = FALSE)
  out$node_label <- factor(out$node_label, levels = factor_levels); out
}

efigure6_data <- do.call(rbind,lapply(GROUP_ORDER, make_efigure6_data))
efigure6_limits <- add_padding(efigure6_data$value, proportion = 0.06, minimum = 0.04)

make_efigure6_panel <- function(group_name, panel_letter) {
  d <- make_efigure6_data(group_name)
  ggplot(d,aes(x = node_label, y = value, group = 1)) +
    geom_hline(yintercept = 0,linewidth = 0.35,color = "grey55") + geom_line(linewidth = 0.50,color = "grey25") + geom_point(size = 1.55,color = "black") + coord_flip() +
    scale_x_discrete(drop = TRUE) + scale_y_continuous(limits = efigure6_limits,expand = expansion(mult = c(0.01, 0.01))) +
    labs(title = paste0(panel_letter, ". ", GROUP_TITLE[group_name]),x = NULL,y = "Predictor-only bridge expected influence (1-step)") +
    theme_grey(base_size = 9) + theme(plot.title = element_text(face = "bold",size = 10,hjust = 0),axis.text.y = element_text(size = 7.4),axis.text.x = element_text(size = 7.2),axis.title.x = element_text(size = 8.5),plot.margin = margin(5.5, 8, 5.5, 8))
}

efigure6 <- make_efigure6_panel("total", "A") / make_efigure6_panel("male", "B") / make_efigure6_panel("female", "C") + plot_layout(heights = c(1,1,1))

# Figure S7
read_boot_object <- function(group_name, file_name) {
  path <- file.path(RESULT_DIR, group_name, file_name)
  if (!file.exists(path)) stop("Bootstrap 객체가 없습니다: ", path)
  object <- readRDS(path)
  if (!inherits(object, "bootnet")) stop("bootnet 객체가 아닙니다: ", path)
  object
}

edge_boot <- setNames(lapply(GROUP_ORDER, read_boot_object, file_name = "30_bootstrap_edge_object.rds"),GROUP_ORDER)
case_boot <- setNames(lapply(GROUP_ORDER, read_boot_object, file_name = "31_bootstrap_case_all_centralities_object.rds"),GROUP_ORDER)

rename_case_statistics <- function(p) {
  if (!is.null(p$data) && "type" %in% names(p$data)) {
    readable_labels <- c(strength = "Strength",expectedInfluence = "Expected influence",bridgeExpectedInfluence = "Bridge expected influence")
    current_type <- as.character(p$data$type)
    p$data$type <- factor(current_type,levels = names(readable_labels),labels = unname(readable_labels))
  }
  p
}

make_edge_boot_panel <- function(group_name, panel_letter) {
  p <- plot(edge_boot[[group_name]],statistics = "edge",plot = "area",labels = FALSE,order = "sample",legend = TRUE,panels = TRUE)
  if (!inherits(p, "ggplot")) stop("Edge bootstrap plot이 ggplot 객체로 생성되지 않았습니다: ", group_name)
  p + labs(title = paste0(panel_letter, ". ", GROUP_TITLE[group_name])) + theme(plot.title = element_text(face = "bold",size = 10,hjust = 0),legend.position = "top",legend.title = element_blank(),legend.text = element_text(size = 6.5),axis.title.x = element_text(size = 8),axis.title.y = element_blank(),axis.text.x = element_text(size = 6.5),axis.text.y = element_blank(),axis.ticks.y = element_blank(),plot.margin = margin(5.5, 5.5, 5.5, 5.5))
}

make_case_boot_panel <- function(group_name, panel_letter) {
  p <- plot(case_boot[[group_name]],statistics = c("strength","expectedInfluence","bridgeExpectedInfluence"),plot = "area",legend = TRUE,panels = TRUE,perNode = FALSE,subsetRange = c(100, 0))
  if (!inherits(p, "ggplot")) stop("Case-dropping plot이 ggplot 객체로 생성되지 않았습니다: ", group_name)
  p <- rename_case_statistics(p)
  p + labs(title = paste0(panel_letter, ". ", GROUP_TITLE[group_name])) + coord_cartesian(ylim = c(0.50, 1.02)) + theme(plot.title = element_text(face = "bold",size = 10,hjust = 0),legend.position = "top",legend.title = element_blank(),legend.text = element_text(size = 6.5),axis.title = element_text(size = 8),axis.text = element_text(size = 6.5),plot.margin = margin(5.5, 5.5, 5.5, 5.5))
}

edge_panels <- list(make_edge_boot_panel("total", "A"),make_edge_boot_panel("male", "B"),make_edge_boot_panel("female", "C"))
case_panels <- list(make_case_boot_panel("total", "D"),make_case_boot_panel("male", "E"),make_case_boot_panel("female", "F"))
efigure7 <- ((edge_panels[[1]] | edge_panels[[2]] | edge_panels[[3]]) / (case_panels[[1]] | case_panels[[2]] | case_panels[[3]])) + plot_layout(guides = "collect",heights = c(1, 1.05)) & theme(legend.position = "top")

save_pdf_png <- function(plot_object, stem, width, height, dpi = 400) {
  pdf_path <- file.path(OUTPUT_DIR, paste0(stem, ".pdf")); png_path <- file.path(OUTPUT_DIR, paste0(stem, ".png"))
  ggsave(filename = pdf_path,plot = plot_object,width = width,height = height,units = "in",device = "pdf",limitsize = FALSE)
  ggsave(filename = png_path,plot = plot_object,width = width,height = height,units = "in",dpi = dpi,bg = "white",limitsize = FALSE)
}

save_pdf_png(efigure5,"Figure_S5_Centrality_Indices",width = 8.5,height = 12.0)
save_pdf_png(efigure6,"Figure_S6_Predictor_Only_Bridge_EI",width = 7.5,height = 11.5)
save_pdf_png(efigure7,"Figure_S7_Bootstrap_Accuracy_Stability",width = 15.5,height = 9.5)

write.csv(efigure5_data,file.path(OUTPUT_DIR, "Figure_S5_plot_data.csv"),row.names = FALSE,fileEncoding = "UTF-8")
write.csv(efigure6_data,file.path(OUTPUT_DIR, "Figure_S6_plot_data.csv"),row.names = FALSE,fileEncoding = "UTF-8")
capture.output(sessionInfo(),file = file.path(OUTPUT_DIR, "plotting_session_info.txt"))

cat("\n============================================================\n")
cat("eFIGURES 5-7 RECREATED FROM FINAL SAVED RESULTS\n")
cat("Bootstrap was NOT rerun.\n")
cat("Output folder:\n", normalizePath(OUTPUT_DIR, winslash = "/"), "\n", sep = "")
cat("\nCreated files:\n")
cat("- Figure_S5_Centrality_Indices.pdf/.png\n")
cat("- Figure_S6_Predictor_Only_Bridge_EI.pdf/.png\n")
cat("- Figure_S7_Bootstrap_Accuracy_Stability.pdf/.png\n")
cat("============================================================\n")
