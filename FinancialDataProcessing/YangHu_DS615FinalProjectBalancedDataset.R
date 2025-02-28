# Load necessary libraries
if (!require(dplyr)) install.packages('dplyr'); library(dplyr)
if (!require(rpart)) install.packages('rpart'); library(rpart)
if (!require(randomForest)) install.packages('randomForest'); library(randomForest)
if (!require(cluster)) install.packages('cluster'); library(cluster)
if (!require(FSelector)) install.packages('FSelector'); library(FSelector)
if (!require(mclust)) install.packages('mclust'); library(mclust)
if (!require(caret)) install.packages('caret'); library(caret)

# Load dataset
balanced_df <- read.csv("~/balanced_dataset.csv") %>% select(-trans_num, -first, -last, -street, -Unnamed..0, -cc_num, -unix_time)

# Remove rows with missing values
# Split 'trans_date_trans_time' into 'date' and 'time' columns
balanced_df$trans_date <- as.Date(balanced_df$trans_date_trans_time)
balanced_df$trans_time <- format(as.POSIXct(balanced_df$trans_date_trans_time), format = "%H:%M:%S")

# Remove original 'trans_date_trans_time' column
balanced_df <- balanced_df %>% select(-trans_date_trans_time)

# Remove rows with missing values
balanced_df <- na.omit(balanced_df)

# Ensure 'is_fraud' is a factor for classification
balanced_df$is_fraud <- as.factor(balanced_df$is_fraud)

# Feature selection using Information Gain
info_gain <- information.gain(is_fraud ~ ., data = balanced_df)
selected_features <- cutoff.k(info_gain, 10) # Select top 10 features

# Create a selected dataset including only the selected features and target 'is_fraud'
selected_df <- balanced_df[, c(selected_features, "is_fraud")]

# Save the selected dataset to a CSV file
write.csv(selected_df, "selected_features_dataset.csv", row.names = FALSE)

# Remove columns with non-numeric data for K-means clustering
# Convert categorical variables to numeric by mapping each category to a unique number
categorical_cols <- sapply(selected_df, is.factor)
selected_df[categorical_cols] <- lapply(selected_df[categorical_cols], function(x) as.numeric(as.factor(x)))

# Create numeric_df from the updated selected_df
numeric_df <- selected_df[, sapply(selected_df, is.numeric)]

# Split the dataset into training and testing sets
set.seed(123) # For reproducibility
train_index <- createDataPartition(selected_df$is_fraud, p = 0.7, list = FALSE)
train_df <- selected_df[train_index, ]
test_df <- selected_df[-train_index, ]

# Create a decision tree model using rpart
set.seed(123) # For reproducibility
fraud_tree <- rpart(is_fraud ~ ., data = train_df, method = "class", control = rpart.control(cp = 0.01, maxdepth = 10))

# Print the summary of the model
printcp(fraud_tree)

# Plot the decision tree with reduced layers
pruned_model <- prune(fraud_tree, cp = fraud_tree$cptable[which.min(fraud_tree$cptable["xerror"]), "CP"])

# Increase the plotting area size and adjust text size
par(xpd = NA, mar = rep(2, 4)) # Increase plot margins
plot(pruned_model, compress = TRUE, margin = 0.2)
text(pruned_model, use.n = TRUE, cex = 0.6)

# Predict using the fraud tree on the test set
tree_predictions <- predict(pruned_model, test_df, type = "class")

# Create a Random Forest model
set.seed(123) # For reproducibility
fraud_rf <- randomForest(is_fraud ~ ., data = train_df, ntree = 100, mtry = 2, importance = TRUE, na.action = na.omit)

# Plot variable importance
varImpPlot(fraud_rf)

# Predict using the Random Forest model on the test set
rf_predictions <- predict(fraud_rf, test_df, type = "class")

# Create K-means clustering model
set.seed(123) # For reproducibility
kmeans_model <- kmeans(scale(numeric_df), centers = 2, nstart = 25)

# Plot K-means clustering with legend and axis labels
clusplot(balanced_df, kmeans_model$cluster, color = TRUE, shade = TRUE, labels = 2, lines = 0, main = "K-means Clustering", xlab = "Component 1", ylab = "Component 2")
legend("topright", legend = unique(kmeans_model$cluster), col = 1:2, pch = 19, title = "Clusters")

# Create Gaussian Mixture Model clustering using Mclust
set.seed(123) # For reproducibility
gmm_model <- Mclust(numeric_df, G = 2)

# Plot GMM clustering results
plot(gmm_model, what = "classification")

# Evaluate model performance on the test set
conf_matrix_tree <- confusionMatrix(factor(tree_predictions, levels = levels(test_df$is_fraud)), test_df$is_fraud)
conf_matrix_rf <- confusionMatrix(factor(rf_predictions, levels = levels(test_df$is_fraud)), test_df$is_fraud)

# Print accuracy of each model
cat("Decision Tree Accuracy:", conf_matrix_tree$overall["Accuracy"], "\n")
cat("Random Forest Accuracy:", conf_matrix_rf$overall["Accuracy"], "\n")

# Print confusion matrices
print(conf_matrix_tree)
print(conf_matrix_rf)

# Save model predictions and clustering results to the dataset
test_df$tree_prediction <- tree_predictions
test_df$rf_prediction <- rf_predictions
test_df$kmeans_cluster <- kmeans_model$cluster[test_index <- as.numeric(rownames(test_df))]
test_df$gmm_cluster <- gmm_model$classification[test_index]

# Save the updated dataset to a CSV file
write.csv(test_df, "test_dataset_with_predictions.csv", row.names = FALSE)
