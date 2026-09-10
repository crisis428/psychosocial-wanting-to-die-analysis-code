# ----------------------------------------------------------------------------
# 10. Bootstrap helpers
# ----------------------------------------------------------------------------

save_boot_table <- function(boot_object, path) {
  if (!is.null(boot_object$bootTable)) write_csv_utf8(as.data.frame(boot_object$bootTable), path)
}
bootstrap_count_range <- function(boot_object, statistics = NULL) {
  tab <- as.data.frame(boot_object$bootTable)
  if (!is.null(statistics) && "type" %in% names(tab)) tab <- tab[tab$type %in% statistics, , drop = FALSE]
  if (nrow(tab) == 0L) return(c(minimum=0L,maximum=0L))
  if (!"id" %in% names(tab)) tab$id <- if ("node2" %in% names(tab)) paste(tab$node1,tab$node2,sep="--") else as.character(tab$node1)
  counts <- table(paste(tab$type,tab$id,sep="::")); c(minimum=min(as.integer(counts)),maximum=max(as.integer(counts)))
}
make_edge_ci_summary <- function(boot_object) {
  boot_tab <- as.data.frame(boot_object$bootTable); sample_tab <- as.data.frame(boot_object$sampleTable)
  boot_tab <- boot_tab[boot_tab$type=="edge",,drop=FALSE]; sample_tab <- sample_tab[sample_tab$type=="edge",,drop=FALSE]
  if (nrow(boot_tab)==0L) return(data.frame())
  boot_tab$edge_id <- paste(boot_tab$node1,boot_tab$node2,sep="--"); split_values <- split(boot_tab$value,boot_tab$edge_id)
  ci <- do.call(rbind,lapply(names(split_values),function(edge_id){ x<-split_values[[edge_id]]; nodes<-strsplit(edge_id,"--",fixed=TRUE)[[1]]; data.frame(edge_id=edge_id,node_1=nodes[1],node_2=nodes[2],bootstrap_mean=mean(x,na.rm=TRUE),bootstrap_sd=stats::sd(x,na.rm=TRUE),ci_2_5=unname(stats::quantile(x,.025,na.rm=TRUE,type=6)),bootstrap_median=unname(stats::quantile(x,.50,na.rm=TRUE,type=6)),ci_97_5=unname(stats::quantile(x,.975,na.rm=TRUE,type=6)),nonzero_in_bootstrap_proportion=mean(abs(x)>sqrt(.Machine$double.eps),na.rm=TRUE),n_successful_replicates=sum(is.finite(x)))}))
  sample_tab$edge_id <- paste(sample_tab$node1,sample_tab$node2,sep="--"); original <- sample_tab[,c("edge_id","value"),drop=FALSE]; names(original)[2] <- "original_edge_weight"
  out <- merge(ci,original,by="edge_id",all.x=TRUE,sort=FALSE); out$absolute_original_edge_weight <- abs(out$original_edge_weight); out <- out[order(out$absolute_original_edge_weight,decreasing=TRUE),]; rownames(out)<-NULL; out
}
safe_cor_stability <- function(boot_object, statistic, group_name) {
  result <- tryCatch(bootnet::corStability(boot_object,cor=.7,statistics=statistic,verbose=FALSE),error=function(e) structure(NA_real_,names=statistic,error_message=conditionMessage(e)))
  data.frame(group=group_name,statistic=if(is.null(names(result))) statistic else names(result),correlation_threshold=.7,cs_coefficient=as.numeric(result),error_message=if(is.null(attr(result,"error_message"))) "" else attr(result,"error_message"))
}
save_boot_plot <- function(path_pdf,path_png,plot_function,width=10,height=7) {
  pdf(path_pdf,width=width,height=height,useDingbats=FALSE); plot_function(); dev.off()
  png(path_png,width=width*400,height=height*400,res=400,bg="white"); plot_function(); dev.off()
}
