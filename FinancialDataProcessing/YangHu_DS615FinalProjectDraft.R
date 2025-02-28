credit_card_transactions <- read.csv("~/Documents/Day1CPT/Courses/DS615-DataMiningAndWarehousing/FinalProject/credit_card_transactions.csv")
  View(credit_card_transactions)
summary(credit_card_transactions)

# Load necessary libraries
install.packages("dplyr")
library(dplyr)

df <- credit_card_transactions

# Select rows where 'is_fraud' is TRUE
fraud_df <- df %>%
  filter(is_fraud == TRUE)

# Select an equal number of rows where 'is_fraud' is FALSE
non_fraud_df <- df %>%
  filter(is_fraud == FALSE) %>%
  sample_n(nrow(fraud_df))

# Combine the two data frames
balanced_df <- bind_rows(fraud_df, non_fraud_df)

# Shuffle the rows of the combined data frame
balanced_df <- balanced_df %>%
  sample_frac(1)

# Save the resulting dataset to a CSV file
write.csv(balanced_df, "balanced_dataset.csv", row.names = FALSE)









# Count the number of rows that have is_fraud == 1 
sum(balanced_df$is_fraud == 1)
# Count the number of rows that have is_fraud == 0
sum(balanced_df$is_fraud == 0)

install.packages("dplyr")
install.packages("ggplot2")
install.packages("gridExtra")

# Load necessary libraries
library(dplyr)
library(ggplot2)
# View the first few rows of the dataset
head(balanced_df)

# Convert is_fraud to a factor since it's a classification problem
balanced_df$is_fraud <- as.factor(balanced_df$is_fraud)
data <- balanced_df

# Convert to Date-Time format
data$trans_date_trans_time <- as.POSIXct(data$trans_date_trans_time, format = "%Y-%m-%d %H:%M:%S")

# Extract relevant features like hour, day of the week, etc.
data$trans_hour <- as.numeric(format(data$trans_date_trans_time, "%H"))
data$trans_day <- as.factor(weekdays(data$trans_date_trans_time))

# Drop the original column
data$trans_date_trans_time <- NULL


# Removing rows with missing values
data_clean <- na.omit(data)
head(data_clean)

# Count of target variable
is_fraud_distribution <- data_clean %>%
  group_by(is_fraud) %>%
  summarize(count = n())

# Print the distribution
is_fraud_distribution

# Install necessary packages
install.packages("rpart")      # for decision tree
install.packages("e1071")      # for SVM
install.packages("pROC")       # for ROC curve
install.packages("caret")      # for training the models

# Load required libraries
library(rpart)
library(e1071)
library(pROC)
library(caret)
# Split the data into training and testing sets
set.seed(567)

head(data_clean)

# Split train and test
trainIndex <- createDataPartition(data_clean$is_fraud, p = .7, list = FALSE)
train <- data_clean[trainIndex, ]
test <- data_clean[-trainIndex, ]

write.csv(train,"~/h2otrain.csv", row.names = FALSE)
write.csv(test,"~/h2otest.csv", row.names = FALSE)

### Try Decision tree
# Install necessary packages
install.packages("rpart")
install.packages("rpart.plot")

# Load the libraries
library(rpart)
library(rpart.plot)


# Build the decision tree model
# fraud_tree <- rpart(is_fraud ~ ., data = train, method = "class")
# Create a decision tree model using rpart
set.seed(123) # For reproducibility
fraud_tree <- rpart(is_fraud ~ ., data = train, method = "class", control = rpart.control(cp = 0.01, maxdepth = 10))

# Print the summary of the model
printcp(fraud_tree)

# Plot the decision tree with reduced layers
pruned_model <- pruned_model <- prune(fraud_tree, cp = fraud_tree$cptable[which.min(fraud_tree$cptable["xerror"]), "CP"])

# Increase the plotting area size and adjust text size
par(xpd = NA, mar = rep(2, 4)) # Increase plot margins
plot(pruned_model, compress = TRUE, margin = 0.2)
text(pruned_model, use.n = TRUE, cex = 0.6)
# Plot the decision tree
# rpart.plot(fraud_tree, type = 3, extra = 102, under = TRUE, cex = 0.1)


# Summary of the model
summary(fraud_tree)

# Variable importance
fraud_tree$variable.importance

train_data <- train
test_data <- test

# Make predictions on the test set
predictions <- predict(fraud_tree, test_data, type = "class")

# Confusion matrix to evaluate model performance
conf_matrix <- table(Predicted = predictions, Actual = test_data$is_fraud)
print(conf_matrix)

# Calculate accuracy
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
cat("Accuracy:", accuracy, "\n")

# Additional metrics like precision, recall, and F1-score
library(caret)
confusionMatrix(predictions, test_data$is_fraud)




## Random Forest Model
# Install necessary packages
install.packages("randomForest")
install.packages("caret")

# Load the libraries
library(randomForest)
library(caret)

train_data <- train
test_data <- test

# Build the Random Forest model
set.seed(123)  # For reproducibility
fraud_rf <- randomForest(is_fraud ~ ., data = train_data, ntree = 500, mtry = sqrt(ncol(train_data) - 1), importance = TRUE)

# Summary of the model
print(fraud_rf)

# Variable importance plot
importance(fraud_rf)
varImpPlot(fraud_rf)

# Make predictions on the test set
predictions <- predict(fraud_rf, test_data)

# Confusion matrix to evaluate model performance
conf_matrix <- confusionMatrix(predictions, test_data$is_fraud)
print(conf_matrix)

# Additional metrics: accuracy, precision, recall, F1-score
cat("Accuracy:", conf_matrix$overall['Accuracy'], "\n")
cat("Precision:", conf_matrix$byClass['Pos Pred Value'], "\n")
cat("Recall:", conf_matrix$byClass['Sensitivity'], "\n")
cat("F1 Score:", 2 * (conf_matrix$byClass['Pos Pred Value'] * conf_matrix$byClass['Sensitivity']) / 
      (conf_matrix$byClass['Pos Pred Value'] + conf_matrix$byClass['Sensitivity']), "\n")









if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }
pkgs <- c("RCurl","jsonlite")
for (pkg in pkgs) {
  if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkg) }
}
install.packages("h2o", type="source", repos=(c("http://h2o-release.s3.amazonaws.com/h2o/latest_stable_R")))
library(h2o)
install.packages("devtools")
library(devtools)
install_github("h2oai/h2o-3/h2o-r/ensemble/h2oEnsemble-package")

# Start the H2O cluster (locally)
h2o.init()





trainIndex <- createDataPartition(data_clean$is_fraud, p = .7, list = FALSE)
train <- data_clean[trainIndex, ]
test <- data_clean[-trainIndex, ]

write.csv(train,"~/h2otrain.csv", row.names = FALSE)
write.csv(test,"~/h2otest.csv", row.names = FALSE)
train <- h2o.importFile("/Users/yanghu/h2otrain.csv")
test <- h2o.importFile("/Users/yanghu/h2otest.csv")


## Try H2O autoML
# Identify predictors and response
y <- "is_fraud"
x <- setdiff(names(train), y)

# For binary classification, response should be a factor
train[, y] <- as.factor(train[, y])
test[, y] <- as.factor(test[, y])



# Run deeplearning
aml <- h2o.deeplearning(x = x, y = y,
                  training_frame = train,
                  validation_frame = test,
                  seed = 1)


# Run AutoML for 20 base models
aml <- h2o.automl(x = x, y = y,
                  training_frame = train,
                  max_models = 20,
                  seed = 1)

# View the AutoML Leaderboard
lb <- aml@leaderboard
print(lb, n = nrow(lb))  # Print all rows instead of default (6 rows)
