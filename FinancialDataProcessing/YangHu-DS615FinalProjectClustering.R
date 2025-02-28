# Load necessary libraries
if (!require(dplyr)) install.packages('dplyr'); library(dplyr)
if (!require(cluster)) install.packages('cluster'); library(cluster)
if (!require(FSelector)) install.packages('FSelector'); library(FSelector)
if (!require(clustMixType)) install.packages('clustMixType'); library(clustMixType)
if (!require(dbscan)) install.packages('dbscan'); library(dbscan)
if (!require(caret)) install.packages('caret'); library(caret)

# Load dataset
balanced_df <- read.csv("~/balanced_dataset.csv") %>% select(-trans_num, -first, -last, -street, -Unnamed..0, -cc_num, -unix_time)

# Remove rows with missing values
balanced_df <- na.omit(balanced_df)

# Calculate age from 'dob' and add 'age' column
# Set buckets for 'amt'
balanced_df$amt_bucket <- cut(balanced_df$amt, breaks = c(-Inf, 50, 100, 500, 1000, Inf), labels = c("0-50", "51-100", "101-500", "501-1000", "1001+"))
current_year <- as.numeric(format(Sys.Date(), "%Y"))
balanced_df$age <- current_year - as.numeric(format(as.Date(balanced_df$dob, "%Y-%m-%d"), "%Y"))

# Convert 'age' to age bucket

# Categorize 'city' and 'job'
balanced_df$city <- as.factor(balanced_df$city)
balanced_df$job <- as.factor(balanced_df$job)
levels(balanced_df$city) <- paste0("City_Group_", as.numeric(levels(balanced_df$city)))
levels(balanced_df$job) <- paste0("Job_Group_", as.numeric(levels(balanced_df$job)))
balanced_df$age_bucket <- cut(balanced_df$age, breaks = c(0, 18, 35, 50, 65, Inf), labels = c("0-18", "19-35", "36-50", "51-65", "65+"))

# Split 'trans_date_trans_time' into 'month' and 'date of the month' columns
balanced_df$trans_month <- format(as.Date(balanced_df$trans_date_trans_time), "%m")
balanced_df$trans_date_of_the_month <- format(as.Date(balanced_df$trans_date_trans_time), "%d")


# Remove original 'trans_date_trans_time' column and trans_date
balanced_df <- balanced_df %>% select(-trans_date_trans_time, -dob, -age, -amt, -merchant)

# Remove rows with missing values
balanced_df <- na.omit(balanced_df)

# Ensure 'is_fraud' is a factor for classification
balanced_df$is_fraud <- as.factor(balanced_df$is_fraud)

# Feature selection using Information Gain
info_gain <- information.gain(is_fraud ~ ., data = balanced_df)
selected_features <- cutoff.k(info_gain, 10) # Select top 10 features

# Categorize all imported numerical features
numeric_cols <- sapply(balanced_df, is.numeric)
balanced_df[numeric_cols] <- lapply(balanced_df[numeric_cols], function(x) {
  cut(x, breaks = quantile(x, probs = seq(0, 1, 0.2), na.rm = TRUE), include.lowest = TRUE, labels = FALSE)
})

# Create a selected dataset including only the selected features and target 'is_fraud'
selected_df <- balanced_df[, c(selected_features, "is_fraud")]

# Save the selected dataset to a CSV file
write.csv(selected_df, "selected_features_dataset.csv", row.names = FALSE)

# Split the dataset into training and testing sets
set.seed(123) # For reproducibility
train_index <- createDataPartition(selected_df$is_fraud, p = 0.6, list = FALSE)
train_df <- selected_df[train_index, ]

# Convert character columns to factors to ensure compatibility with Gower distance calculation
train_df[] <- lapply(train_df, function(x) if (is.character(x)) as.factor(x) else x)
train_df[] <- lapply(names(train_df), function(col) {
  if (is.factor(test_df[[col]])) {
    factor(train_df[[col]], levels = union(levels(train_df[[col]]), levels(test_df[[col]])))
  } else {
    train_df[[col]]
  }
})

# Ensure consistent factor levels between training and testing sets
test_df <- selected_df[-train_index, ]
test_df[] <- lapply(names(test_df), function(col) {
  if (is.factor(train_df[[col]])) {
    factor(test_df[[col]], levels = union(levels(train_df[[col]]), levels(test_df[[col]])))
  } else {
    test_df[[col]]
  }
})
test_df <- as.data.frame(test_df)
train_df[] <- lapply(train_df, function(x) if (is.character(x)) as.factor(x) else x)
test_df <- selected_df[-train_index, ]

# Convert character columns to factors to ensure compatibility with Gower distance calculation
test_df[] <- lapply(test_df, function(x) if (is.character(x)) as.factor(x) else x)

# K-Prototypes Clustering
set.seed(123) # For reproducibility
kproto_model <- kproto(train_df, k = 2)
kproto_clusters <- predict(kproto_model, test_df)$cluster
kproto_clusters <- as.factor(kproto_clusters)

# Hierarchical Clustering with Gower Distance
gower_dist <- daisy(train_df, metric = "gower")
hc_model <- hclust(gower_dist, method = "ward.D2")
hc_clusters <- cutree(hc_model, k = 2)

# DBSCAN with Gower Distance
gower_dist_test <- daisy(test_df, metric = "gower")
dbscan_model <- dbscan(as.matrix(gower_dist_test), eps = 0.5, minPts = 5)
dbscan_clusters <- dbscan_model$cluster

# Evaluate clustering performance
cat("K-Prototypes Clustering Results:\n")
print(table(kproto_clusters, factor(test_df$is_fraud, levels = levels(kproto_clusters))))

cat("Hierarchical Clustering Results:\n")
print(table(hc_clusters, train_df$is_fraud))

cat("DBSCAN Clustering Results:\n")
print(table(dbscan_clusters, test_df$is_fraud))

# Save clustering results to the test dataset
test_df$kproto_cluster <- kproto_clusters
test_df$hc_cluster <- hc_clusters[as.numeric(rownames(test_df))]
test_df$dbscan_cluster <- dbscan_clusters

# Save the updated dataset to a CSV file
write.csv(test_df, "test_dataset_with_predictions.csv", row.names = FALSE)

