# Stage 6: R Shiny Research Dashboard
# This app reads the clean dashboard CSV files created in Stage 5.

library(shiny)

APP_VERSION <- "research_layer_stage06_shiny_dashboard_v1"

DASHBOARD_FILES <- c(
  "dashboard_kpis.csv",
  "model_metric_long_table.csv",
  "model_ranking_table.csv",
  "family_summary_table.csv",
  "best_model_summary_table.csv",
  "confusion_summary_table.csv",
  "prediction_distribution_table.csv",
  "top_feature_importance_table.csv",
  "feature_importance_by_model_table.csv",
  "top_organisations_table.csv"
)


app_folder <- function() {
  normalizePath(dirname(sys.frame(1)$ofile %||% getwd()), winslash = "/")
}


`%||%` <- function(left_value, right_value) {
  if (is.null(left_value)) {
    return(right_value)
  }
  
  left_value
}


dashboard_data_folder <- function() {
  file.path(getwd(), "shiny_dashboard", "dashboard_data")
}


local_data_folder <- function() {
  file.path(getwd(), "dashboard_data")
}


find_data_folder <- function() {
  candidate_folders <- c(dashboard_data_folder(), local_data_folder())
  valid_folders <- candidate_folders[sapply(candidate_folders, has_dashboard_files)]
  
  if (length(valid_folders) == 0) {
    stop("Could not find shiny_dashboard/dashboard_data files.")
  }
  
  normalizePath(valid_folders[[1]], winslash = "/")
}


has_dashboard_files <- function(folder_path) {
  all(file.exists(file.path(folder_path, DASHBOARD_FILES)))
}


read_csv_safely <- function(file_path) {
  tryCatch(
    read.csv(file_path, stringsAsFactors = FALSE, fileEncoding = "UTF-8-BOM"),
    error = function(error_details) {
      stop(paste("Could not read:", file_path, error_details$message))
    }
  )
}


read_dashboard_table <- function(data_folder, file_name) {
  read_csv_safely(file.path(data_folder, file_name))
}


load_dashboard_data <- function() {
  data_folder <- find_data_folder()
  
  list(
    kpis = read_dashboard_table(data_folder, "dashboard_kpis.csv"),
    metrics_long = read_dashboard_table(data_folder, "model_metric_long_table.csv"),
    ranking = read_dashboard_table(data_folder, "model_ranking_table.csv"),
    family = read_dashboard_table(data_folder, "family_summary_table.csv"),
    best = read_dashboard_table(data_folder, "best_model_summary_table.csv"),
    confusion = read_dashboard_table(data_folder, "confusion_summary_table.csv"),
    prediction = read_dashboard_table(data_folder, "prediction_distribution_table.csv"),
    features = read_dashboard_table(data_folder, "top_feature_importance_table.csv"),
    features_by_model = read_dashboard_table(data_folder, "feature_importance_by_model_table.csv"),
    organisations = read_dashboard_table(data_folder, "top_organisations_table.csv")
  )
}


number_percent <- function(value) {
  paste0(round(as.numeric(value) * 100, 2), "%")
}


lookup_kpi <- function(kpi_table, metric_name) {
  matched_value <- kpi_table$value[kpi_table$metric == metric_name]
  
  if (length(matched_value) == 0) {
    return("Missing")
  }
  
  matched_value[[1]]
}


safe_numeric <- function(value) {
  suppressWarnings(as.numeric(value))
}


metric_label <- function(metric_name) {
  tools::toTitleCase(gsub("_", " ", metric_name))
}


clean_table_names <- function(data_table) {
  names(data_table) <- tools::toTitleCase(gsub("_", " ", names(data_table)))
  data_table
}


top_rows <- function(data_table, row_limit) {
  if (nrow(data_table) == 0) {
    return(data_table)
  }
  
  head(data_table, row_limit)
}


sort_by_column_desc <- function(data_table, column_name) {
  data_table[[column_name]] <- safe_numeric(data_table[[column_name]])
  data_table[order(-data_table[[column_name]]), , drop = FALSE]
}


prepare_ranking_table <- function(ranking_table) {
  wanted_columns <- c("overall_rank", "display_model_name", "method_family", "f1_score", "auc", "recall")
  clean_table_names(ranking_table[wanted_columns])
}


prepare_family_table <- function(family_table) {
  clean_table_names(family_table)
}


prepare_organisation_table <- function(organisation_table) {
  wanted_columns <- c("rank_overall", "organisation_name", "influence_score", "influence_band")
  clean_table_names(organisation_table[wanted_columns])
}


prepare_feature_table <- function(feature_table) {
  clean_table_names(feature_table)
}


metric_choices <- function(metrics_table) {
  unique(metrics_table$metric_name)
}


model_choices <- function(ranking_table) {
  ranking_table$display_model_name
}


confusion_choices <- function(confusion_table) {
  confusion_table$display_model_name
}


find_model_key <- function(ranking_table, display_name) {
  ranking_table$model_key[ranking_table$display_model_name == display_name][[1]]
}


filter_metric_table <- function(metrics_table, selected_metric) {
  metrics_table[metrics_table$metric_name == selected_metric, , drop = FALSE]
}


filter_model_metrics <- function(metrics_table, selected_key) {
  metrics_table[metrics_table$model_key == selected_key, , drop = FALSE]
}


filter_confusion_model <- function(confusion_table, selected_name) {
  confusion_table[confusion_table$display_model_name == selected_name, , drop = FALSE]
}


dashboard_css <- function() {
  HTML("
    body { background: #f4f6fb; }
    .main-title { font-size: 30px; font-weight: 800; color: #152238; }
    .subtitle { color: #5d6678; margin-bottom: 20px; }
    .kpi-card {
      background: white; padding: 18px; border-radius: 16px;
      box-shadow: 0 4px 14px rgba(0,0,0,0.08); min-height: 110px;
    }
    .kpi-title { color: #697386; font-size: 13px; font-weight: 700; }
    .kpi-value { color: #111827; font-size: 24px; font-weight: 800; }
    .panel-card {
      background: white; padding: 20px; border-radius: 16px;
      box-shadow: 0 4px 14px rgba(0,0,0,0.08); margin-bottom: 18px;
    }
    .section-title { font-size: 20px; font-weight: 800; color: #152238; }
  ")
}


kpi_card <- function(title, value) {
  div(
    class = "kpi-card",
    div(class = "kpi-title", title),
    div(class = "kpi-value", value)
  )
}


dashboard_header <- function() {
  div(
    class = "panel-card",
    div(class = "main-title", "AI Ecosystem Intelligence Research Dashboard"),
    div(class = "subtitle", "Machine Learning, Deep Learning, Influence Modelling, and Organisation Analytics")
  )
}


overview_tab <- function() {
  tabPanel(
    "Overview",
    dashboard_header(),
    fluidRow(
      column(3, uiOutput("best_model_card")),
      column(3, uiOutput("best_f1_card")),
      column(3, uiOutput("best_auc_card")),
      column(3, uiOutput("real_model_card"))
    ),
    br(),
    fluidRow(
      column(7, div(class = "panel-card", plotOutput("ranking_plot", height = "430px"))),
      column(5, div(class = "panel-card", tableOutput("top_ranking_table")))
    )
  )
}


comparison_tab <- function() {
  tabPanel(
    "ML vs DL",
    div(class = "panel-card", div(class = "section-title", "Family-Level Comparison")),
    fluidRow(
      column(6, div(class = "panel-card", plotOutput("family_f1_plot", height = "360px"))),
      column(6, div(class = "panel-card", tableOutput("family_table")))
    )
  )
}


metric_explorer_tab <- function(data_bundle) {
  tabPanel(
    "Metric Explorer",
    div(class = "panel-card", selectInput("selected_metric", "Choose metric", metric_choices(data_bundle$metrics_long))),
    fluidRow(
      column(8, div(class = "panel-card", plotOutput("metric_plot", height = "440px"))),
      column(4, div(class = "panel-card", tableOutput("metric_table")))
    )
  )
}


model_detail_tab <- function(data_bundle) {
  tabPanel(
    "Model Detail",
    div(class = "panel-card", selectInput("selected_model", "Choose model", model_choices(data_bundle$ranking))),
    fluidRow(
      column(6, div(class = "panel-card", plotOutput("single_model_plot", height = "360px"))),
      column(6, div(class = "panel-card", tableOutput("single_model_table")))
    )
  )
}


confusion_tab <- function(data_bundle) {
  tabPanel(
    "Confusion Matrix",
    div(class = "panel-card", selectInput("selected_confusion_model", "Choose model", confusion_choices(data_bundle$confusion))),
    fluidRow(
      column(6, div(class = "panel-card", plotOutput("confusion_plot", height = "360px"))),
      column(6, div(class = "panel-card", tableOutput("confusion_table")))
    )
  )
}


features_tab <- function() {
  tabPanel(
    "Feature Importance",
    fluidRow(
      column(7, div(class = "panel-card", plotOutput("feature_plot", height = "430px"))),
      column(5, div(class = "panel-card", tableOutput("feature_table")))
    )
  )
}


organisations_tab <- function() {
  tabPanel(
    "Top Organisations",
    fluidRow(
      column(7, div(class = "panel-card", plotOutput("organisation_plot", height = "430px"))),
      column(5, div(class = "panel-card", tableOutput("organisation_table")))
    )
  )
}


app_ui <- function(data_bundle) {
  fluidPage(
    tags$head(tags$style(dashboard_css())),
    tabsetPanel(
      overview_tab(),
      comparison_tab(),
      metric_explorer_tab(data_bundle),
      model_detail_tab(data_bundle),
      confusion_tab(data_bundle),
      features_tab(),
      organisations_tab()
    )
  )
}


plot_horizontal_bars <- function(values, labels, title_text, x_label) {
  par(mar = c(5, 12, 4, 2))
  barplot(
    rev(values),
    names.arg = rev(labels),
    horiz = TRUE,
    las = 1,
    main = title_text,
    xlab = x_label
  )
}


ranking_plot_data <- function(ranking_table) {
  real_models <- ranking_table[ranking_table$model_type == "model", , drop = FALSE]
  sort_by_column_desc(real_models, "f1_score")
}


metric_plot_data <- function(metrics_table, selected_metric) {
  filtered_table <- filter_metric_table(metrics_table, selected_metric)
  sort_by_column_desc(filtered_table, "metric_value")
}


feature_plot_data <- function(feature_table) {
  sorted_table <- sort_by_column_desc(feature_table, "importance")
  top_rows(sorted_table, 15)
}


organisation_plot_data <- function(organisation_table) {
  sorted_table <- sort_by_column_desc(organisation_table, "influence_score")
  top_rows(sorted_table, 15)
}


render_kpi_cards <- function(output, data_bundle) {
  output$best_model_card <- renderUI(kpi_card("Best Model", lookup_kpi(data_bundle$kpis, "best_model")))
  output$best_f1_card <- renderUI(kpi_card("Best F1-Score", number_percent(lookup_kpi(data_bundle$kpis, "best_model_f1_score"))))
  output$best_auc_card <- renderUI(kpi_card("Best AUC", number_percent(lookup_kpi(data_bundle$kpis, "best_model_auc"))))
  output$real_model_card <- renderUI(kpi_card("Real Models", lookup_kpi(data_bundle$kpis, "real_model_count")))
}


render_overview_outputs <- function(output, data_bundle) {
  output$ranking_plot <- renderPlot(plot_ranking(data_bundle$ranking))
  output$top_ranking_table <- renderTable(prepare_ranking_table(top_rows(data_bundle$ranking, 10)))
}


plot_ranking <- function(ranking_table) {
  plot_table <- ranking_plot_data(ranking_table)
  plot_horizontal_bars(plot_table$f1_score, plot_table$display_model_name, "Overall Model Ranking by F1-Score", "F1-score")
}


render_family_outputs <- function(output, data_bundle) {
  output$family_f1_plot <- renderPlot(plot_family_summary(data_bundle$family))
  output$family_table <- renderTable(prepare_family_table(data_bundle$family))
}


plot_family_summary <- function(family_table) {
  family_table$best_f1_score <- safe_numeric(family_table$best_f1_score)
  barplot(family_table$best_f1_score, names.arg = family_table$method_family, main = "Best F1-Score by Method Family", ylab = "F1-score")
}


render_metric_outputs <- function(output, input, data_bundle) {
  output$metric_plot <- renderPlot(plot_selected_metric(data_bundle$metrics_long, input$selected_metric))
  output$metric_table <- renderTable(metric_table_output(data_bundle$metrics_long, input$selected_metric))
}


plot_selected_metric <- function(metrics_table, selected_metric) {
  plot_table <- metric_plot_data(metrics_table, selected_metric)
  plot_horizontal_bars(plot_table$metric_value, plot_table$display_model_name, metric_label(selected_metric), "Metric value")
}


metric_table_output <- function(metrics_table, selected_metric) {
  plot_table <- metric_plot_data(metrics_table, selected_metric)
  clean_table_names(top_rows(plot_table, 12))
}


render_single_model_outputs <- function(output, input, data_bundle) {
  output$single_model_plot <- renderPlot(plot_single_model(data_bundle, input$selected_model))
  output$single_model_table <- renderTable(single_model_table(data_bundle, input$selected_model))
}


plot_single_model <- function(data_bundle, selected_model) {
  selected_key <- find_model_key(data_bundle$ranking, selected_model)
  model_table <- filter_model_metrics(data_bundle$metrics_long, selected_key)
  barplot(model_table$metric_value, names.arg = metric_label(model_table$metric_name), las = 2, main = selected_model)
}


single_model_table <- function(data_bundle, selected_model) {
  selected_key <- find_model_key(data_bundle$ranking, selected_model)
  model_table <- filter_model_metrics(data_bundle$metrics_long, selected_key)
  clean_table_names(model_table)
}


render_confusion_outputs <- function(output, input, data_bundle) {
  output$confusion_plot <- renderPlot(plot_confusion_model(data_bundle$confusion, input$selected_confusion_model))
  output$confusion_table <- renderTable(clean_table_names(filter_confusion_model(data_bundle$confusion, input$selected_confusion_model)))
}


plot_confusion_model <- function(confusion_table, selected_model) {
  row <- filter_confusion_model(confusion_table, selected_model)
  values <- c(row$true_positive, row$false_positive, row$true_negative, row$false_negative)
  barplot(values, names.arg = c("TP", "FP", "TN", "FN"), main = selected_model, ylab = "Count")
}


render_feature_outputs <- function(output, data_bundle) {
  output$feature_plot <- renderPlot(plot_features(data_bundle$features))
  output$feature_table <- renderTable(prepare_feature_table(top_rows(data_bundle$features, 15)))
}


plot_features <- function(feature_table) {
  plot_table <- feature_plot_data(feature_table)
  plot_horizontal_bars(plot_table$importance, plot_table$feature_name, "Top Feature Importance", "Importance")
}


render_organisation_outputs <- function(output, data_bundle) {
  output$organisation_plot <- renderPlot(plot_organisations(data_bundle$organisations))
  output$organisation_table <- renderTable(prepare_organisation_table(data_bundle$organisations))
}


plot_organisations <- function(organisation_table) {
  plot_table <- organisation_plot_data(organisation_table)
  plot_horizontal_bars(plot_table$influence_score, plot_table$organisation_name, "Top Organisations by Influence", "Influence score")
}


app_server <- function(data_bundle) {
  function(input, output, session) {
    render_kpi_cards(output, data_bundle)
    render_overview_outputs(output, data_bundle)
    render_family_outputs(output, data_bundle)
    render_metric_outputs(output, input, data_bundle)
    render_single_model_outputs(output, input, data_bundle)
    render_confusion_outputs(output, input, data_bundle)
    render_feature_outputs(output, data_bundle)
    render_organisation_outputs(output, data_bundle)
  }
}


build_sample_metric_table <- function() {
  data.frame(model_name = c("a", "b"), metric_name = c("f1_score", "f1_score"), metric_value = c(0.5, 0.9))
}


test_sunny_day_percent <- function() {
  # Test 1: percent formatting should convert decimal scores properly.
  stopifnot(number_percent(0.923077) == "92.31%")
}


test_edge_case_kpi_lookup <- function() {
  # Test 2: missing KPI values should not crash the dashboard.
  stopifnot(lookup_kpi(data.frame(metric = "x", value = "1"), "missing") == "Missing")
}


test_weird_gotcha_metric_label <- function() {
  # Test 3: metric labels should be readable in the dashboard.
  stopifnot(metric_label("balanced_accuracy") == "Balanced Accuracy")
}


test_bug_catcher_sorting <- function() {
  # Test 4: sorting should place the highest metric first.
  table <- sort_by_column_desc(build_sample_metric_table(), "metric_value")
  stopifnot(table$model_name[[1]] == "b")
}


run_self_tests <- function() {
  test_sunny_day_percent()
  test_edge_case_kpi_lookup()
  test_weird_gotcha_metric_label()
  test_bug_catcher_sorting()
}


run_self_tests()

dashboard_data <- load_dashboard_data()
shinyApp(ui = app_ui(dashboard_data), server = app_server(dashboard_data))
