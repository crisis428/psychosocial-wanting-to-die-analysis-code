# ----------------------------------------------------------------------------
# 11. Nonparametric edge bootstrap and case-dropping stability
# ----------------------------------------------------------------------------
bootstrap_log <- list(); cs_tables <- list()
for (group_name in names(network_results)) {
  result <- network_results[[group_name]]; group_dir <- result$group_dir
  message("\n--- Bootstrap: ", group_name, " ---")
  full_communities <- NODE_DOMAIN[result$variables]; full_communities[OUTCOME] <- "Focal"
  predictor_domains <- c("Psychological","Subjective health","Social-contextual","Demographic")
  edge_checkpoint <- file.path(group_dir,"30_bootstrap_edge_nonparametric.rds"); edge_warning_file <- file.path(group_dir,"30_bootstrap_edge_warnings.txt")
  edge_boot <- read_checkpoint(edge_checkpoint)
  if (!is.null(edge_boot)) edge_warnings <- if (file.exists(edge_warning_file)) readLines(edge_warning_file) else character(0) else {
    set.seed(result$seed+100L)
    edge_run <- run_and_capture_warnings(bootnet::bootnet(result$estimate,nBoots=N_BOOT_EDGE,nCores=N_CORES,type="nonparametric",statistics="edge",memorysaver=TRUE,verbose=TRUE,maxErrors=20))
    edge_boot <- edge_run$value; edge_warnings <- edge_run$warnings; save_checkpoint(edge_boot,edge_checkpoint); writeLines(edge_warnings,edge_warning_file)
  }
  case_checkpoint <- file.path(group_dir,"31_bootstrap_case_all_centralities.rds"); case_warning_file <- file.path(group_dir,"31_bootstrap_case_all_centralities_warnings.txt")
  case_boot <- read_checkpoint(case_checkpoint)
  if (!is.null(case_boot)) case_warnings <- if (file.exists(case_warning_file)) readLines(case_warning_file) else character(0) else {
    set.seed(result$seed+200L)
    case_run <- run_and_capture_warnings(bootnet::bootnet(result$estimate,nBoots=N_BOOT_CASE,nCores=N_CORES,type="case",statistics=c("strength","expectedInfluence","bridgeExpectedInfluence"),communities=full_communities,useCommunities=predictor_domains,bridgeArgs=list(normalize=FALSE),caseMin=CASE_MIN,caseMax=CASE_MAX,caseN=CASE_LEVELS,memorysaver=TRUE,verbose=TRUE,maxErrors=20))
    case_boot <- case_run$value; case_warnings <- case_run$warnings; save_checkpoint(case_boot,case_checkpoint); writeLines(case_warnings,case_warning_file)
  }
  save_boot_table(edge_boot,file.path(group_dir,"30_bootstrap_edge_table.csv")); save_boot_table(case_boot,file.path(group_dir,"31_bootstrap_case_all_centralities_table.csv"))
  saveRDS(edge_boot,file.path(group_dir,"30_bootstrap_edge_object.rds")); saveRDS(case_boot,file.path(group_dir,"31_bootstrap_case_all_centralities_object.rds"))
  edge_ci_summary <- make_edge_ci_summary(edge_boot); write_csv_utf8(edge_ci_summary,file.path(group_dir,"30_bootstrap_edge_CI_summary.csv"))
  edge_count_range <- bootstrap_count_range(edge_boot,"edge"); case_count_range <- bootstrap_count_range(case_boot,c("strength","expectedInfluence","bridgeExpectedInfluence"))
  save_boot_plot(file.path(group_dir,"Figure_bootstrap_edge_accuracy.pdf"),file.path(group_dir,"Figure_bootstrap_edge_accuracy.png"),function(){print(plot(edge_boot,labels=FALSE,order="sample"))},width=11,height=8)
  save_boot_plot(file.path(group_dir,"Figure_case_drop_strength_EI.pdf"),file.path(group_dir,"Figure_case_drop_strength_EI.png"),function(){print(plot(case_boot,statistics=c("strength","expectedInfluence")))},width=9,height=7)
  save_boot_plot(file.path(group_dir,"Figure_case_drop_bridge_EI.pdf"),file.path(group_dir,"Figure_case_drop_bridge_EI.png"),function(){print(plot(case_boot,statistics="bridgeExpectedInfluence"))},width=9,height=7)
  cs_tables[[paste0(group_name,"_strength")]] <- safe_cor_stability(case_boot,"strength",group_name)
  cs_tables[[paste0(group_name,"_ei")]] <- safe_cor_stability(case_boot,"expectedInfluence",group_name)
  cs_tables[[paste0(group_name,"_bridge")]] <- safe_cor_stability(case_boot,"bridgeExpectedInfluence",group_name)
  bootstrap_log[[group_name]] <- data.frame(group=group_name,edge_bootstraps_requested=N_BOOT_EDGE,case_bootstraps_requested=N_BOOT_CASE,edge_successful_replicates_min=unname(edge_count_range["minimum"]),edge_successful_replicates_max=unname(edge_count_range["maximum"]),case_successful_replicates_min=unname(case_count_range["minimum"]),case_successful_replicates_max=unname(case_count_range["maximum"]),edge_warning_count=length(edge_warnings),case_warning_count=length(case_warnings),edge_warnings=paste(edge_warnings,collapse=" | "),case_warnings=paste(case_warnings,collapse=" | "))
}
cs_all <- do.call(rbind,cs_tables); rownames(cs_all)<-NULL; write_csv_utf8(cs_all,file.path(output_dir,"40_CS_coefficients_all_groups.csv"))
bootstrap_log_all <- do.call(rbind,bootstrap_log); rownames(bootstrap_log_all)<-NULL; write_csv_utf8(bootstrap_log_all,file.path(output_dir,"41_bootstrap_run_log.csv"))
# ----------------------------------------------------------------------------
# 12. Reproducibility manifest and final object bundle
# ----------------------------------------------------------------------------
settings_table <- data.frame(setting=c("checkpoint_tag","test_mode","source_file","source_md5","EBIC_gamma","n_lambda","lambda_min_ratio","post_EBIC_edge_filter","edge_bootstraps","case_bootstraps","case_drop_min","case_drop_max","case_drop_levels","cores","seed","layout","layout_rule","layout_distance_preserving","positive_edge","negative_edge"),value=c(CHECKPOINT_TAG,TEST_MODE,data_path,source_md5,EBIC_GAMMA,N_LAMBDA,LAMBDA_MIN_RATIO,"none",N_BOOT_EDGE,N_BOOT_CASE,CASE_MIN,CASE_MAX,CASE_LEVELS,N_CORES,SEED,paste0("predictor-only spring layout ","estimated separately by group"),paste0("global PCA orientation/reflection and ","intentional vertical compression; ","no individual predictor-node relocation; ","outcome fixed above predictor layer"),paste0("no; vertical compression is an intentional ","visualization transform"),"red solid","blue solid"))
write_csv_utf8(settings_table,file.path(output_dir,"50_analysis_settings.csv"))
package_versions <- data.frame(package=required_packages,version=vapply(required_packages,function(package) as.character(utils::packageVersion(package)),character(1)))
write_csv_utf8(package_versions,file.path(output_dir,"50_package_versions.csv"))
saveRDS(list(settings=settings_table,data_qc=qc,node_dictionary=node_dictionary,common_edge_scale=common_edge_scale,networks=lapply(network_results,function(x){x$data<-NULL;x}),centrality=centrality_all,cs_coefficients=cs_all,bootstrap_log=bootstrap_log_all),file.path(output_dir,"51_network_analysis_complete_objects.rds"))
capture.output(sessionInfo(),file=file.path(output_dir,"99_session_info_final.txt"))
cat("\n============================================================\n"); cat("NETWORK ANALYSIS COMPLETED\n"); cat("Mode: ",ifelse(TEST_MODE,"TEST","FINAL"),"\n",sep=""); cat("Output: ",normalizePath(output_dir,winslash="/"),"\n",sep=""); cat("N: total = ",network_results$total$n,"; male = ",network_results$male$n,"; female = ",network_results$female$n,"\n",sep=""); cat("Common edge scale maximum: ",common_edge_scale,"\n",sep=""); cat("No post-EBICglasso edge filter was applied.\n"); cat("============================================================\n")
