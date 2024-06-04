#load the csv file 
library(dplyr)
library(tidyr)
library(lme4)
library(lmerTest)
library(ggplot2)
data <- read.csv("/Users/venturelab/Documents/git-papers/nv_rest_network/nv_rest_network/within_network_means_with_covariates.csv", header = TRUE)
data <- data.frame(data)
# Assuming 'network' is a factor with 13 levels
data_long_rest$network <- factor(data_long_rest$network)



#model trial with interaction 

model1_extended <- lmer(value ~ age * network + Gender + hash + alcohol + (session | subject), data = data_long_rest)
summary(model1_extended)

# Ensure the network column is a factor
data_long_rest$network <- as.factor(data_long_rest$network)

# Set sum-to-zero contrasts for the network factor
contrasts(data_long_rest$network) <- contr.sum(length(levels(data_long_rest$network)))

# Fit the extended model
model1_extended <- lmer(value ~ age * network + Gender + hash + alcohol + (session | subject), data = data_long_rest)

# Summarize the model
summary(model1_extended)


#plot the data to visualize network trajectories

ggplot(data_long_rest, aes(x = age, y = value, color = network)) +
  geom_point(alpha = 0.3) +  # Make points more transparent
  geom_smooth(method = "lm", se = FALSE) +
  facet_wrap(~ network, scales = "free_y") +
  theme(legend.position = "bottom") +
  labs(title = "Individual Network Values Across Age",
       x = "Age",
       y = "Value",
       color = "Network")



