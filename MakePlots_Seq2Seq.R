#This script requires the following packages:
library(dplyr)
library(ggplot2)
library(entropy)

sem <- function (that_data){
  s <- sd(that_data)
  n <- length(that_data)
  error <- s/sqrt(n)
  
  return(error)
}


#Find averages and error bars:
this_data <- read.csv("Curves by trial type (NN-ForcedChoice).csv", header=T)
grouped_data <- group_by(this_data, Pattern, Trial.Type, Epoch)
summarized <- summarize(grouped_data, mean(Accuracy), sem(Accuracy))

#Run t-tests on final epoch's results:
bleed_pal <- subset(this_data, Trial.Type == "Palatalizing" & Pattern == "Bleeding" & Epoch == 19)$Accuracy
bleed_inter <- subset(this_data, Trial.Type == "Interacting" & Pattern == "Bleeding" & Epoch == 19)$Accuracy
feed_pal <- subset(this_data, Trial.Type == "Palatalizing" & Pattern == "Feeding" & Epoch == 19)$Accuracy
feed_inter <- subset(this_data, Trial.Type == "Interacting" & Pattern == "Feeding" & Epoch == 19)$Accuracy
cb_pal <- subset(this_data, Trial.Type == "Palatalizing" & Pattern == "Counterbleeding" & Epoch == 19)$Accuracy
cb_inter <- subset(this_data, Trial.Type == "Interacting" & Pattern == "Counterbleeding" & Epoch == 19)$Accuracy
cf_pal <- subset(this_data, Trial.Type == "Palatalizing" & Pattern == "Counterfeeding" & Epoch == 19)$Accuracy
cf_inter <- subset(this_data, Trial.Type == "Interacting" & Pattern == "Counterfeeding" & Epoch == 19)$Accuracy

#Test for Transparency bias:
inter_transp <- c(bleed_inter, feed_inter)
inter_opaque <- c(cb_inter, cf_inter)
t.test(inter_transp, inter_opaque, var.equal=TRUE)

#Test for MaxUtil bias:
pal_util <- c(cb_pal, feed_pal)
pal_faith <- c(cf_pal, bleed_pal)
t.test(pal_util, pal_faith, var.equal=TRUE)


#Plot curves:
summarized$Epoch <- summarized$Epoch + 1
interacting <- subset(summarized, summarized$Trial.Type=="Interacting")
ggplot(interacting, aes(x=Epoch, y=`mean(Accuracy)`, linetype=Pattern))+
  geom_ribbon(aes(ymin=`mean(Accuracy)`-`sem(Accuracy)`,
                  ymax=`mean(Accuracy)`+`sem(Accuracy)`),fill="white", color="black", alpha=1.0)+
  geom_point(aes(shape=Pattern), size=3)+
  geom_line(size=1)+
  theme(text = element_text(size=20))+
  ylim(.3,1)+
  xlim(1,20)+
  labs(x="Iteration", y="Average Accuracy", title="Interacting Items (Seq2Seq)", subtitle="White region shows standard error of the mean")

palatalizing <- subset(summarized, summarized$Trial.Type=="Palatalizing")
ggplot(palatalizing, aes(x=Epoch, y=`mean(Accuracy)`, linetype=Pattern))+
  geom_ribbon(aes(ymin=`mean(Accuracy)`-`sem(Accuracy)`,
                  ymax=`mean(Accuracy)`+`sem(Accuracy)`),fill="white", color="black", alpha=1.0)+
  geom_line(size=1)+
  geom_point(aes(shape=Pattern), size=3)+
  theme(text = element_text(size=20))+
  ylim(.3,1)+
  xlim(1,20)+
  labs(x="Iteration", y="Average Accuracy", title="Palatalizing Items (Seq2Seq)", subtitle="White region shows standard error of the mean")

#Plot bars at end of learning:
end <- subset(summarized, summarized$Epoch == 19)
#my_colors <- c("#ff9999","#cc99ff","cornflowerblue","white")#Color figure
my_colors <- c("grey48", "gray26", "grey82", "white") #B&W figure (in paper)
end$Trial.Type <- factor(end$Trial.Type, levels = c("Faithful", "Deleting", "Palatalizing", "Interacting"))
end$Pattern <- factor(end$Pattern, levels = c("Bleeding", "Feeding", "Counterbleeding", "Counterfeeding"))
ggplot(end, aes(x=Trial.Type, y=`mean(Accuracy)`, fill=Pattern))+
  geom_col(position="dodge", color="black") +
  ylab("Average Accuracy") + 
  xlab("Trial Type") +
  ylim(c(0,1))+
  theme(text = element_text(size=30))+
  scale_fill_manual(values=c(my_colors, my_colors), name="Interaction Type")+
  geom_errorbar(aes(ymin=`mean(Accuracy)`-`sem(Accuracy)`, ymax=`mean(Accuracy)`+`sem(Accuracy)`, width=.2), position=position_dodge(.9), alpha=1)+
  labs(title="Accuracy by Pattern and Trial Type for Seq2Seq Learner (at end of learning)",
       subtitle="Error bars show standard error of the mean")

