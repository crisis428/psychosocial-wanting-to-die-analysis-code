# ----------------------------------------------------------------------------
# 9. Centrality outputs and supplementary figures
# ----------------------------------------------------------------------------

centrality_all <- do.call(rbind, lapply(network_results, `[[`, "centrality"))
rownames(centrality_all) <- NULL
write_csv_utf8(centrality_all, file.path(output_dir, "20_centrality_all_groups.csv"))
centrality_long <- do.call(rbind, lapply(split(centrality_all, centrality_all$group), function(d) {
  rbind(data.frame(group=d$group,node=d$node,node_code=d$node_code,metric="Strength",raw_value=d$strength,standardized_value=z_score_safe(d$strength)), data.frame(group=d$group,node=d$node,node_code=d$node_code,metric="Expected influence",raw_value=d$expected_influence_1step,standardized_value=z_score_safe(d$expected_influence_1step)), data.frame(group=d$group,node=d$node,node_code=d$node_code,metric="Bridge expected influence",raw_value=d$bridge_expected_influence_1step_predictor_only,standardized_value=z_score_safe(d$bridge_expected_influence_1step_predictor_only)))
}))
centrality_long <- centrality_long[!is.na(centrality_long$raw_value), , drop=FALSE]
write_csv_utf8(centrality_long, file.path(output_dir, "21_centrality_long.csv"))
CENTRALITY_ORDER_TOP_TO_BOTTOM <- c(OUTCOME,"pss14_total","poor_subjective_physical_health","poor_subjective_mental_health","married_current","low_social_trust","low_sense_of_belonging","low_self_esteem_score","low_perceived_social_equality","living_alone","gad7_total","sex_female","employed_corrected","age_years")
NODE_PLOT_LABEL <- c(py_thoughts_wanting_to_die="Thoughts of wanting to die",gad7_total="GAD-7 score",pss14_total="PSS-14 score",low_self_esteem_score="Low self-esteem",poor_subjective_physical_health="Poor subjective physical health",poor_subjective_mental_health="Poor subjective mental health",low_sense_of_belonging="Low sense of belonging",low_perceived_social_equality="Low perceived social equality",low_social_trust="Low social trust",age_years="Age",sex_female="Female sex",married_current="Married",employed_corrected="Employed",living_alone="Living alone")
group_label_map <- c(total="A. Total sample",male="B. Male subgroup",female="C. Female subgroup")
group_label_levels <- unname(group_label_map[c("total","male","female")])
node_factor_levels <- rev(unname(NODE_PLOT_LABEL[CENTRALITY_ORDER_TOP_TO_BOTTOM]))
centrality_s2 <- centrality_long[centrality_long$metric %in% c("Strength","Expected influence"), , drop=FALSE]
centrality_s2$node_label <- factor(unname(NODE_PLOT_LABEL[centrality_s2$node]),levels=node_factor_levels)
centrality_s2$group_label <- factor(unname(group_label_map[centrality_s2$group]),levels=group_label_levels)
centrality_s2$metric <- factor(centrality_s2$metric,levels=c("Strength","Expected influence"))
p_centrality_s2 <- ggplot(centrality_s2,aes(x=node_label,y=standardized_value,group=1))+geom_hline(yintercept=0,linewidth=.35,color="grey55")+geom_line(linewidth=.50,color="grey25")+geom_point(size=1.45,color="black")+coord_flip()+facet_grid(rows=vars(group_label),cols=vars(metric),switch="y",drop=FALSE)+labs(x=NULL,y="Standardized centrality (z score)")+theme_grey(base_size=9)+theme(strip.text=element_text(face="bold",size=9),strip.text.y.left=element_text(angle=0,hjust=1),strip.placement="outside",axis.text.y=element_text(size=7.3),panel.spacing.y=grid::unit(.8,"lines"),legend.position="none")
ggsave(file.path(output_dir,"Supplementary_Figure_S2_Centrality.pdf"),p_centrality_s2,width=8.5,height=12.5,device="pdf")
ggsave(file.path(output_dir,"Supplementary_Figure_S2_Centrality.png"),p_centrality_s2,width=8.5,height=12.5,dpi=400,bg="white")
centrality_s3 <- centrality_long[centrality_long$metric=="Bridge expected influence", , drop=FALSE]
if (any(centrality_s3$node==OUTCOME)) stop("The focal outcome must not appear in predictor-only bridge EI.")
centrality_s3$node_label <- factor(unname(NODE_PLOT_LABEL[centrality_s3$node]),levels=node_factor_levels)
centrality_s3$group_label <- factor(unname(group_label_map[centrality_s3$group]),levels=group_label_levels)
p_centrality_s3 <- ggplot(centrality_s3,aes(x=node_label,y=raw_value,group=1))+geom_hline(yintercept=0,linewidth=.35,color="grey55")+geom_line(linewidth=.50,color="grey25")+geom_point(size=1.45,color="black")+coord_flip()+facet_grid(rows=vars(group_label),switch="y",drop=FALSE)+labs(x=NULL,y=paste0("Predictor-only bridge expected ","influence (1-step)"))+theme_grey(base_size=9)+theme(strip.text=element_text(face="bold",size=9),strip.text.y.left=element_text(angle=0,hjust=1),strip.placement="outside",axis.text.y=element_text(size=7.3),panel.spacing.y=grid::unit(.8,"lines"),legend.position="none")
ggsave(file.path(output_dir,"Supplementary_Figure_S3_Bridge_EI.pdf"),p_centrality_s3,width=7.5,height=12.5,device="pdf")
ggsave(file.path(output_dir,"Supplementary_Figure_S3_Bridge_EI.png"),p_centrality_s3,width=7.5,height=12.5,dpi=400,bg="white")
