#!/usr/bin/env Rscript

# Step 1: Data loading and cleaning for the research modelling layer.
# This script prepares Dataset 3 organisation influence data for R ML models.

SCRIPT_VERSION <- "research_layer_step01_data_loading_cleaning_v1"

DEFAULT_INPUT_PATHS <- c(
  "data/analytics/Dataset3_Organisation_Influence_Scores.csv",
  "data/processed/Dataset3_Organisation_Influence_Scores.csv",
  "Dataset3_Organisation_Influence_Scores.csv"
)

OUTPUT_FOLDER <- "research_layer/outputs"

REQUIRED_COLUMNS <- c(
  "canonical_entity_id",
  "canonical_name",
  "primary_platform",
  "influence_score",
  "influence_band",
  "artefact_count",
  "repository_count",
  "model_count",
  "dataset_count",
  "degree_total",
  "weighted_degree_total",
  "pagerank_score",
  "bridge_score"
)

LEAKAGE_COLUMNS <- c(
  "rank_overall",
  "influence_score",
  "influence_band"
)


find_input_file <- function(candidate_paths) {
  existing_paths <- candidate_paths[file.exists(candidate_paths)]
  if (length(existing_paths) == 0) {
    stop("No Dataset 3 organisation influence file was found.")
  }
  existing_paths[[1]]
}


load_research_data <- function(file_path) {
  tryCatch(
    read.csv(file_path, stringsAsFactors = FALSE, fileEncoding = "UTF-8-BOM"),
    error = function(read_error) {
      stop(paste("Could not read input file:", read_error$message))
    }
  )
}


validate_required_columns <- function(research_data) {
  missing_columns <- setdiff(REQUIRED_COLUMNS, names(research_data))
  if (length(missing_columns) > 0) {
    stop(paste("Missing required columns:", paste(missing_columns, collapse = ", ")))
  }
  TRUE
}


clean_text_column <- function(raw_column) {
  cleaned_column <- trimws(as.character(raw_column))
  cleaned_column[is.na(cleaned_column)] <- "unknown"
  cleaned_column[cleaned_column == ""] <- "unknown"
  cleaned_column
}


clean_number_column <- function(raw_column) {
  numeric_text <- gsub(",", "", as.character(raw_column))
  numeric_values <- suppressWarnings(as.numeric(numeric_text))
  numeric_values[is.na(numeric_values)] <- 0
  numeric_values
}


make_binary_target <- function(influence_band) {
  clean_band <- tolower(trimws(as.character(influence_band)))
  ifelse(clean_band %in% c("medium", "high"), 1, 0)
}


clean_research_data <- function(research_data) {
  validate_required_columns(research_data)
  research_data$canonical_entity_id <- clean_text_column(research_data$canonical_entity_id)
  research_data$canonical_name <- clean_text_column(research_data$canonical_name)
  research_data$primary_platform <- clean_text_column(research_data$primary_platform)
  research_data$influence_band_clean <- clean_text_column(research_data$influence_band)
  research_data$medium_or_high_influence <- make_binary_target(research_data$influence_band)
  clean_numeric_columns(research_data)
}


clean_numeric_columns <- function(research_data) {
  numeric_columns <- find_numeric_columns(research_data)
  for (column_name in numeric_columns) {
    research_data[[column_name]] <- clean_number_column(research_data[[column_name]])
  }
  research_data
}


find_numeric_columns <- function(research_data) {
  possible_columns <- c(
    "influence_score", "artefact_count", "repository_count",
    "model_count", "dataset_count", "degree_total",
    "weighted_degree_total", "pagerank_score", "bridge_score"
  )
  intersect(possible_columns, names(research_data))
}


build_feature_table <- function(cleaned_data) {
  numeric_columns <- names(cleaned_data)[sapply(cleaned_data, is.numeric)]
  predictor_columns <- setdiff(numeric_columns, LEAKAGE_COLUMNS)
  predictor_columns <- setdiff(predictor_columns, "medium_or_high_influence")
  feature_table <- cleaned_data[c("canonical_entity_id", predictor_columns)]
  feature_table$medium_or_high_influence <- cleaned_data$medium_or_high_influence
  feature_table
}


build_cleaning_report <- function(cleaned_data, input_path) {
  data.frame(
    metric = report_metric_names(),
    value = report_metric_values(cleaned_data, input_path),
    script_version = SCRIPT_VERSION,
    stringsAsFactors = FALSE
  )
}


report_metric_names <- function() {
  c(
    "input_file",
    "row_count",
    "column_count",
    "positive_target_count",
    "negative_target_count",
    "unique_platform_count"
  )
}


report_metric_values <- function(cleaned_data, input_path) {
  c(
    input_path,
    nrow(cleaned_data),
    ncol(cleaned_data),
    sum(cleaned_data$medium_or_high_influence == 1),
    sum(cleaned_data$medium_or_high_influence == 0),
    length(unique(cleaned_data$primary_platform))
  )
}


ensure_output_folder <- function(output_folder) {
  if (!dir.exists(output_folder)) {
    dir.create(output_folder, recursive = TRUE)
  }
}


write_csv_safely <- function(data_table, file_path) {
  tryCatch(
    write.csv(data_table, file_path, row.names = FALSE),
    error = function(write_error) {
      stop(paste("Could not write output file:", write_error$message))
    }
  )
}


write_research_outputs <- function(cleaned_data, feature_table, cleaning_report) {
  ensure_output_folder(OUTPUT_FOLDER)
  write_csv_safely(cleaned_data, file.path(OUTPUT_FOLDER, "dataset3_clean_research_table.csv"))
  write_csv_safely(feature_table, file.path(OUTPUT_FOLDER, "dataset3_model_feature_table.csv"))
  write_csv_safely(cleaning_report, file.path(OUTPUT_FOLDER, "dataset3_cleaning_report.csv"))
}


run_step_one <- function() {
  input_path <- find_input_file(DEFAULT_INPUT_PATHS)
  raw_data <- load_research_data(input_path)
  cleaned_data <- clean_research_data(raw_data)
  feature_table <- build_feature_table(cleaned_data)
  cleaning_report <- build_cleaning_report(cleaned_data, input_path)
  write_research_outputs(cleaned_data, feature_table, cleaning_report)
  print("Step 1 complete: cleaned research datasets were written.")
}


test_sunny_day_case <- function() {
  sample_data <- build_sample_data()
  cleaned_data <- clean_research_data(sample_data)
  stopifnot(nrow(cleaned_data) == 2)
  print("PASS: Sunny day test passed because normal rows clean correctly.")
}


test_edge_case_empty_strings <- function() {
  sample_data <- build_sample_data()
  sample_data$primary_platform[1] <- ""
  cleaned_data <- clean_research_data(sample_data)
  stopifnot(cleaned_data$primary_platform[1] == "unknown")
  print("PASS: Edge case test passed because empty text becomes unknown.")
}


test_weird_gotcha_numbers <- function() {
  sample_data <- build_sample_data()
  sample_data$artefact_count[1] <- "1,200"
  cleaned_data <- clean_research_data(sample_data)
  stopifnot(cleaned_data$artefact_count[1] == 1200)
  print("PASS: Weird gotcha test passed because comma numbers are cleaned.")
}


test_bug_catcher_leakage <- function() {
  sample_data <- build_sample_data()
  feature_table <- build_feature_table(clean_research_data(sample_data))
  stopifnot(!"influence_score" %in% names(feature_table))
  print("PASS: Bug catcher test passed because target leakage was removed.")
}


build_sample_data <- function() {
  data.frame(
    canonical_entity_id = c("CORG_001", "CORG_002"),
    canonical_name = c("microsoft", "research lab"),
    primary_platform = c("github", "huggingface"),
    influence_score = c("79.2", "66.1"),
    influence_band = c("high", "medium"),
    artefact_count = c("10", "8"),
    repository_count = c("5", "2"),
    model_count = c("2", "4"),
    dataset_count = c("1", "1"),
    degree_total = c("50", "40"),
    weighted_degree_total = c("100.5", "90.2"),
    pagerank_score = c("0.12", "0.10"),
    bridge_score = c("0.8", "0.6"),
    rank_overall = c("1", "2"),
    stringsAsFactors = FALSE
  )
}


run_self_tests <- function() {
  test_sunny_day_case()
  test_edge_case_empty_strings()
  test_weird_gotcha_numbers()
  test_bug_catcher_leakage()
}


if (sys.nframe() == 0) {
  run_self_tests()
  run_step_one()
}


