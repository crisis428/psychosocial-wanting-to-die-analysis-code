# ----------------------------------------------------------------------------
# 8. Common edge scale and layered network figures
# ----------------------------------------------------------------------------

common_edge_max <- max(vapply(network_results, function(x) max(abs(x$weights[upper.tri(x$weights)])), numeric(1)))
common_edge_scale <- ceiling(common_edge_max * 20) / 20
if (common_edge_scale <= 0) common_edge_scale <- 0.05
edge_scale_table <- data.frame(observed_maximum_absolute_edge = common_edge_max, plotting_scale_maximum = common_edge_scale, positive_edge_color = POSITIVE_EDGE_COLOR, negative_edge_color = NEGATIVE_EDGE_COLOR, positive_edge_line = "solid", negative_edge_line = "solid", edge_filter_after_EBICglasso = "none")
write_csv_utf8(edge_scale_table, file.path(output_dir, "05_common_edge_scale.csv"))
NETWORK_X_LIMITS <- c(-1.58, 1.58); NETWORK_Y_LIMITS <- c(-0.78, 1.35)
plot_one_network <- function(result, panel_title = NULL) {
  vars <- result$variables; node_sizes <- ifelse(vars == OUTCOME, 7.8, 6.4)
  qgraph::qgraph(result$weights, layout = result$layout, labels = unname(NODE_CODE[vars]), color = unname(DOMAIN_COLOR[NODE_DOMAIN[vars]]), directed = FALSE, posCol = POSITIVE_EDGE_COLOR, negCol = NEGATIVE_EDGE_COLOR, negDashed = FALSE, fade = TRUE, trans = TRUE, minimum = 0, cut = 0, threshold = 0, maximum = common_edge_scale, esize = 10, vsize = node_sizes, shape = "circle", label.cex = 0.90, label.color = "white", borders = TRUE, border.color = "white", border.width = 1.2, rescale = FALSE, aspect = TRUE, mar = c(0.5,0.5,2.2,0.5), legend = FALSE, title = panel_title)
}
plot_network_legend <- function() {
  plot.new(); par(usr = c(0,1,0,1), xpd = NA)
  text(0.02,0.98,"Node domains",adj=0,font=2,cex=.88)
  domain_order <- c("Outcome","Demographic","Psychological","Social-contextual","Subjective health"); domain_y <- .94-(seq_along(domain_order)-1)*.035
  for (i in seq_along(domain_order)) { domain_name <- domain_order[i]; points(.055,domain_y[i],pch=21,bg=DOMAIN_COLOR[domain_name],col="white",cex=1.45); text(.11,domain_y[i],domain_name,adj=0,cex=.68) }
  text(.02,.755,"Nodes",adj=0,font=2,cex=.88)
  legend_node_order <- c("age_years","employed_corrected","sex_female","living_alone","married_current",OUTCOME,"gad7_total","pss14_total","low_self_esteem_score","low_sense_of_belonging","low_perceived_social_equality","low_social_trust","poor_subjective_mental_health","poor_subjective_physical_health")
  node_y <- .72-(seq_along(legend_node_order)-1)*.025
  for (i in seq_along(legend_node_order)) { variable_name <- legend_node_order[i]; points(.055,node_y[i],pch=21,bg=DOMAIN_COLOR[NODE_DOMAIN[variable_name]],col="white",cex=1.08); text(.105,node_y[i],unname(NODE_CODE[variable_name]),adj=0,font=2,cex=.57); text(.19,node_y[i],unname(NODE_NAME[variable_name]),adj=0,cex=.54) }
  text(.02,.345,"Absolute partial correlation",adj=0,font=2,cex=.73)
  edge_examples <- c(.1,.3,.5,.7); example_x <- c(.04,.52,.04,.52); example_y <- c(.29,.29,.225,.225)
  for (i in seq_along(edge_examples)) { line_width <- max(.7,8*edge_examples[i]/common_edge_scale); segments(example_x[i],example_y[i],example_x[i]+.12,example_y[i],lwd=line_width,col="grey35"); text(example_x[i]+.16,example_y[i],sprintf("%.1f",edge_examples[i]),adj=0,cex=.62) }
  text(.02,.165,"Edges",adj=0,font=2,cex=.78); segments(.04,.115,.19,.115,lwd=3,col=POSITIVE_EDGE_COLOR); text(.23,.115,"Positive conditional association",adj=0,cex=.55); segments(.04,.065,.19,.065,lwd=3,col=NEGATIVE_EDGE_COLOR); text(.23,.065,"Negative conditional association",adj=0,cex=.55)
}
for (group_name in names(network_results)) {
  result <- network_results[[group_name]]; panel_title <- paste0(result$label," (n = ",format(result$n,big.mark=","),")")
  pdf(file.path(result$group_dir,paste0("Figure_network_",group_name,".pdf")),width=8.2,height=5.8,useDingbats=FALSE,bg="white"); plot_one_network(result,panel_title); dev.off()
  png(file.path(result$group_dir,paste0("Figure_network_",group_name,".png")),width=3280,height=2320,res=400,bg="white"); plot_one_network(result,panel_title); dev.off()
}
save_three_panel_network <- function(file_name, device = c("png","pdf")) {
  device <- match.arg(device)
  if (device == "pdf") pdf(file.path(output_dir,file_name),width=10.5,height=14.5,useDingbats=FALSE,bg="white") else png(file.path(output_dir,file_name),width=4200,height=5800,res=400,bg="white")
  on.exit(dev.off(),add=TRUE)
  graphics::layout(matrix(c(1,4,2,4,3,4),nrow=3,byrow=TRUE),widths=c(3.75,1.80),heights=c(1,1,1)); par(mar=c(.5,.5,2.2,.5))
  plot_one_network(network_results$total,paste0("A. Total sample (n = ",format(network_results$total$n,big.mark=","),")")); plot_one_network(network_results$male,paste0("B. Male subgroup (n = ",format(network_results$male$n,big.mark=","),")")); plot_one_network(network_results$female,paste0("C. Female subgroup (n = ",format(network_results$female$n,big.mark=","),")")); par(mar=c(0,0,0,0)); plot_network_legend()
}
save_three_panel_network("Figure_network_total_male_female_layered.pdf",device="pdf")
save_three_panel_network("Figure_network_total_male_female_layered.png",device="png")
