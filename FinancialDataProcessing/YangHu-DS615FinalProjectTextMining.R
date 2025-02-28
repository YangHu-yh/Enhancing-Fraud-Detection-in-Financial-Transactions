# Load necessary libraries
if (!require(tidyverse)) install.packages("tidyverse")
library(tidyverse)  # For data manipulation and visualization

if (!require(caret)) install.packages("caret")
library(caret)      # For data preprocessing and model training

if (!require(ROSE)) install.packages("ROSE")
library(ROSE)       # For dealing with imbalanced datasets

if (!require(randomForest)) install.packages("randomForest")
library(randomForest) # For Random Forest Model

# Load the dataset
balanced_df <- read.csv("~/balanced_dataset.csv")

# Remove rows with missing values
balanced_df <- na.omit(balanced_df)

# Rename columns to ensure compatibility
colnames(balanced_df) <- make.names(colnames(balanced_df))

# Explore the dataset
str(balanced_df)          # Understand structure of the dataset
summary(balanced_df)      # Summarize statistics of each feature


# Remove rows with missing values
balanced_df <- na.omit(balanced_df)

# Calculate age from 'dob' and add 'age' column
# Set buckets for 'amt'
balanced_df$amt_bucket <- cut(balanced_df$amt, breaks = c(-Inf, 50, 100, 500, 1000, Inf), labels = c("0-50", "51-100", "101-500", "501-1000", "1001+"))
current_year <- as.numeric(format(Sys.Date(), "%Y"))
balanced_df$age <- current_year - as.numeric(format(as.Date(balanced_df$dob, "%Y-%m-%d"), "%Y"))

# Categorize 'city' and 'job'
balanced_df$city <- as.factor(balanced_df$city)
balanced_df$job <- as.factor(balanced_df$job)
levels(balanced_df$city) <- paste0("City_Group_", as.numeric(levels(balanced_df$city)))
levels(balanced_df$job) <- paste0("Job_Group_", as.numeric(levels(balanced_df$job)))
balanced_df$age_bucket <- cut(balanced_df$age, breaks = c(0, 18, 35, 50, 65, Inf), labels = c("0-18", "19-35", "36-50", "51-65", "65+"))

# Split 'trans_date_trans_time' into 'month' and 'date of the month' columns
balanced_df$trans_month <- format(as.Date(balanced_df$trans_date_trans_time), "%m")
balanced_df$trans_date_of_the_month <- format(as.Date(balanced_df$trans_date_trans_time), "%d")


# Save the updated dataset to a CSV file
write.csv(balanced_df, "test_balanced_dataset_with_updated_columns.csv", row.names = FALSE)




# Remove original 'trans_date_trans_time' column and trans_date
balanced_df <- balanced_df %>% select(-trans_date_trans_time, -dob, -age, -amt, -merchant)

# Remove rows with missing values
balanced_df <- na.omit(balanced_df)

# Ensure 'is_fraud' is a factor for classification
balanced_df$is_fraud <- as.factor(balanced_df$is_fraud)

# Categorize all imported numerical features
numeric_cols <- sapply(balanced_df, is.numeric)
balanced_df[numeric_cols] <- lapply(balanced_df[numeric_cols], function(x) {
  cut(x, breaks = quantile(x, probs = seq(0, 1, 0.2), na.rm = TRUE), include.lowest = TRUE, labels = FALSE)
})

# Identify the column name that indicates fraud
fraud_column <- grep("is_fraud", colnames(balanced_df), value = TRUE, ignore.case = TRUE)
if (length(fraud_column) == 0) {
  stop("Error: 'is_fraud' column not found in the balanced_df dataset. Please check the column names.")
} else {
  fraud_column <- fraud_column[1]
}

# Basic EDA: Visualize class distribution
ggplot(balanced_df, aes_string(x = paste0("as.factor(", fraud_column, ")"))) +
  geom_bar(fill = "steelblue") +
  labs(title = "isFraud Distribution", x = "isFraud (0 = Non-Fraud, 1 = Fraud)", y = "Count")

# Visualize numerical feature distributions
num_cols <- balanced_df %>% select(where(is.numeric)) %>% names()
for (col in num_cols) {
  print(ggplot(balanced_df, aes_string(x = col, fill = paste0("as.factor(", fraud_column, ")"))) +
          geom_density(alpha = 0.5) +
          labs(title = paste("Distribution of", col), x = col, fill = "Class"))
}

# Feature Scaling
scaled_data <- balanced_df %>% mutate_if(is.numeric, scale)

# Handle Class Imbalance using ROSE (Random Over-Sampling Examples)
#balanced_data <- ROSE(as.formula(paste(fraud_column, "~ .")), data = scaled_data, seed = 123)$data

# Split the data into training and testing sets
set.seed(123)
train_index <- createDataPartition(scaled_data[[fraud_column]], p = 0.7, list = FALSE)
train_data <- scaled_data[train_index, ]
test_data <- scaled_data[-train_index, ]

# Train a Random Forest model
set.seed(123)
rf_model <- randomForest(as.formula(paste(fraud_column, "~ .")), data = train_data, ntree = 100, mtry = 3, importance = TRUE)

# Evaluate model on test set
rf_predictions <- predict(rf_model, newdata = test_data)
confusion_matrix <- confusionMatrix(rf_predictions, test_data[[fraud_column]], positive = "1")
print(confusion_matrix)

# Variable Importance
varImpPlot(rf_model)

# Precision, Recall, and F1 Score
tp <- confusion_matrix$table[2, 2]
fp <- confusion_matrix$table[1, 2]
fn <- confusion_matrix$table[2, 1]

precision <- tp / (tp + fp)
recall <- tp / (tp + fn)
f1_score <- 2 * (precision * recall) / (precision + recall)

cat("\nPrecision:", precision, "\n")
cat("Recall:", recall, "\n")
cat("F1 Score:", f1_score, "\n")
