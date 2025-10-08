# Clear environment
rm(list = ls(all.names = TRUE))
gc(full = TRUE, verbose = FALSE)

# Load required libraries
require("data.table")
require("parallel")
require("R.utils")
require("primes")
require("utils")
require("rlist")
require("yaml")
require("lightgbm")
require("DiceKriging")
require("mlrMBO")
require("ggplot2")

cat("Libraries loaded successfully\n")
cat("Timestamp:", format(Sys.time(), "%a %b %d %X %Y"), "\n")

# Environment detection and paths
IS_COLAB <- dir.exists("/content/buckets")

if (IS_COLAB) {
  # Colab setup - mount drive first
  # from google.colab import drive
  # drive.mount('/content/.drive')
  
  # Create directory structure
  system("mkdir -p '/content/.drive/My Drive/dmeyf'")
  system("mkdir -p /content/buckets")
  system("ln -s '/content/.drive/My Drive/dmeyf' /content/buckets/b1")
  system("mkdir -p /content/buckets/b1/exp")
  system("mkdir -p /content/buckets/b1/datasets")
  system("mkdir -p /content/datasets")
  
  # Download dataset if needed
  if (!file.exists("/content/buckets/b1/datasets/competencia_01_crudo.csv")) {
    download.file(
      "https://storage.googleapis.com/open-courses/dmeyf2025-e4a2/competencia_01_crudo.csv",
      "/content/buckets/b1/datasets/competencia_01_crudo.csv"
    )
  }
  
  BASE_PATH <- "/content/buckets/b1"
  DATA_PATH <- "/content/datasets/competencia_01_crudo.csv"
} else {
  # Local setup
  setwd('/Users/manumoreira/Repos/dmeyf2025/Competencia1/')
  BASE_PATH <- getwd()
  DATA_PATH <- "./data/competencia_01_crudo.csv"
}

cat("Environment:", ifelse(IS_COLAB, "Colab", "Local"), "\n")
cat("Base path:", BASE_PATH, "\n")
cat("Data path:", DATA_PATH, "\n")

# Experiment parameters
PARAM <- list()
PARAM$experimento <- 8
PARAM$semilla_primigenia <- 450343

# Training periods
PARAM$train <- c(202101, 202102, 202103)
PARAM$train_validate <- c(202101, 202102, 202103)
PARAM$validate <- c(202104)
PARAM$train_final <- c(202101, 202102, 202103, 202104)
PARAM$future <- c(202106)

# Kaggle parameters
PARAM$semilla_kaggle <- 314159
PARAM$cortes <- seq(6000, 19000, by = 500)

# Training strategy
PARAM$trainingstrategy$undersampling <- 0.4

# Hyperparameter tuning
PARAM$hyperparametertuning$xval_folds <- 5
PARAM$hyperparametertuning$iteraciones <- 60

cat("Parameters configured:\n")
cat("  Experiment:", PARAM$experimento, "\n")
cat("  Seed:", PARAM$semilla_primigenia, "\n")
cat("  Training periods:", paste(PARAM$train, collapse = ", "), "\n")
cat("  Validation period:", PARAM$validate, "\n")
cat("  Future period:", PARAM$future, "\n")

# LightGBM fixed parameters
PARAM$lgbm$param_fijos <- list(
  boosting = "gbdt",
  objective = "binary",
  metric = "auc",
  first_metric_only = FALSE,
  boost_from_average = TRUE,
  feature_pre_filter = FALSE,
  force_row_wise = TRUE,
  verbosity = -100,
  seed = PARAM$semilla_primigenia,
  max_depth = -1L,
  min_gain_to_split = 0,
  min_sum_hessian_in_leaf = 0.001,
  lambda_l1 = 0.0,
  lambda_l2 = 0.0,
  max_bin = 31L,
  bagging_fraction = 1.0,
  pos_bagging_fraction = 1.0,
  neg_bagging_fraction = 1.0,
  is_unbalance = FALSE,
  scale_pos_weight = 1.0,
  drop_rate = 0.1,
  max_drop = 50,
  skip_drop = 0.5,
  extra_trees = FALSE,
  num_iterations = 1200,
  learning_rate = 0.02,
  feature_fraction = 0.5,
  num_leaves = 750,
  min_data_in_leaf = 3
)

cat("LightGBM fixed parameters configured\n")

# Hyperparameter search space
PARAM$hypeparametertuning$hs <- makeParamSet(
  makeIntegerParam("num_iterations", lower = 8L, upper = 2048L),
  makeNumericParam("learning_rate", lower = 0.01, upper = 0.3),
  makeNumericParam("feature_fraction", lower = 0.1, upper = 1.0),
  makeIntegerParam("num_leaves", lower = 8L, upper = 2048L),
  makeIntegerParam("lambda_1", lower = 0, upper = 15),
  makeNumericParam("min_gain_to_split", lower = 0.1, upper = 1.0),
  makeIntegerParam("max_depth", lower = 1, upper = 15),
  makeIntegerParam("bagging_freq", lower = 1, upper = 10),
  makeNumericParam("bagging_fraction", lower = 0.1, upper = 1.0)
)

cat("Hyperparameter search space defined\n")
cat("  Number of hyperparameters to tune:", length(PARAM$hypeparametertuning$hs$pars), "\n")

# Load dataset
dataset <- fread(DATA_PATH, stringsAsFactors = TRUE)

cat("Dataset loaded:\n")
cat("  Rows:", nrow(dataset), "\n")
cat("  Columns:", ncol(dataset), "\n")
cat("  Memory:", format(object.size(dataset), units = "MB"), "\n")

# Function to create separate rankings
rank_separate_by_month <- function(dt, columns, month_col = "foto_mes") {
  new_cols <- paste0(columns, "_ranked")
  
  for(i in seq_along(columns)) {
    col <- columns[i]
    new_col <- new_cols[i]
    
    # Initialize with NAs
    dt[, (new_col) := NA_real_]
    
    # Rank within each month
    dt[!is.na(get(col)), (new_col) := {
      # Handle negatives
      neg_vals <- get(col)[get(col) < 0]
      neg_ranks <- if(length(neg_vals) > 0) -frank(-neg_vals) else numeric(0)
      
      # Handle positives
      pos_vals <- get(col)[get(col) > 0]
      pos_ranks <- if(length(pos_vals) > 0) frank(pos_vals) else numeric(0)
      
      # Handle zeros
      zero_vals <- get(col)[get(col) == 0]
      
      # Combine results
      result <- rep(0, length(get(col)))
      result[get(col) < 0] <- neg_ranks
      result[get(col) > 0] <- pos_ranks
      result[get(col) == 0] <- 0
      result
    }, by = get(month_col)]
  }
  
  return(dt)
}

# Apply ranking function
dataset <- rank_separate_by_month(dataset, c("mrentabilidad", "mrentabilidad_annual", "mcomisiones", "mactivos_margen", "mpasivos_margen", 
                "mcuenta_corriente_adicional", "mcuenta_corriente", "mcaja_ahorro", "mcaja_ahorro_adicional", 
                "mcaja_ahorro_dolares", "mcuentas_saldo", "mautoservicio", "mtarjeta_visa_consumo", "mtarjeta_master_consumo", 
                "mprestamos_personales", "mprestamos_prendarios", "mprestamos_hipotecarios", "mplazo_fijo_dolares", 
                "mplazo_fijo_pesos", "minversion1_pesos", "minversion1_dolares", "minversion2", "mpayroll", "mpayroll2", 
                "mcuenta_debitos_automaticos", "mttarjeta_visa_debitos_automaticos", "mttarjeta_master_debitos_automaticos", 
                "mpagodeservicios", "mpagomiscuentas", "mcajeros_propios_descuentos", "mtarjeta_visa_descuentos", 
                "mtarjeta_master_descuentos", "mcomisiones_mantenimiento", "mcomisiones_otras", "mforex_buy", "mforex_sell", 
                "mtransferencias_recibidas", "mtransferencias_emitidas", "mextraccion_autoservicio", "mcheques_depositados", 
                "mcheques_emitidos", "mcheques_depositados_rechazados", "mcheques_emitidos_rechazados", "matm", "matm_other", 
                "Master_mfinanciacion_limite", "Master_msaldototal", "Master_msaldopesos", "Master_msaldodolares", 
                "Master_mconsumospesos", "Master_mconsumosdolares", "Master_mlimitecompra", "Master_madelantopesos", 
                "Master_madelantodolares", "Master_mpagado", "Master_mpagospesos", "Master_mpagosdolares", 
                "Master_mconsumototal", "Master_mpagominimo", "Visa_mfinanciacion_limite", "Visa_msaldototal", 
                "Visa_msaldopesos", "Visa_msaldodolares", "Visa_mconsumospesos", "Visa_mconsumosdolares", 
                "Visa_mlimitecompra", "Visa_madelantopesos", "Visa_madelantodolares", "Visa_mpagado", "Visa_mpagospesos", 
                "Visa_mpagosdolares", "Visa_mconsumototal", "Visa_mpagominimo"))

# Function to create comparison summary
create_compact_monthly_summary <- function(dt, original_cols, month_col = "foto_mes") {
  ranked_cols <- paste0(original_cols, "_ranked")
  
  summary_list <- lapply(seq_along(original_cols), function(i) {
    orig <- original_cols[i]
    rank <- ranked_cols[i]
    
    month_summary <- dt[!is.na(get(month_col)), .(
      # Original stats
      Orig_Median = median(get(orig), na.rm = TRUE),
      Orig_IQR = IQR(get(orig), na.rm = TRUE),
      Orig_Min = min(get(orig), na.rm = TRUE),
      Orig_Max = max(get(orig), na.rm = TRUE),
      
      # Ranked stats
      Rank_Median = median(get(rank), na.rm = TRUE),
      Rank_IQR = IQR(get(rank), na.rm = TRUE),
      Rank_Min = min(get(rank), na.rm = TRUE),
      Rank_Max = max(get(rank), na.rm = TRUE),
      
      # Counts
      Count = .N,
      Neg_Count_Orig = sum(get(orig) < 0, na.rm = TRUE),
      Zero_Count_Orig = sum(get(orig) == 0, na.rm = TRUE),
      Pos_Count_Orig = sum(get(orig) > 0, na.rm = TRUE),
      Neg_Count_Rank = sum(get(rank) < 0, na.rm = TRUE),
      Zero_Count_Rank = sum(get(rank) == 0, na.rm = TRUE),
      Pos_Count_Rank = sum(get(rank) > 0, na.rm = TRUE),
      NA_Count_Orig = sum(is.na(get(orig))),
      NA_Count_Rank = sum(is.na(get(rank)))
    ), by = month_col]
    
    # Add column name
    month_summary[, Column := orig]
    setcolorder(month_summary, c("Column", month_col, names(month_summary)[!names(month_summary) %in% c("Column", month_col)]))
    
    return(month_summary)
  })
  
  return(rbindlist(summary_list))
}

# Usage
original_cols <- c("mrentabilidad", "mrentabilidad_annual", "mcomisiones", "mactivos_margen", "mpasivos_margen", 
                "mcuenta_corriente_adicional", "mcuenta_corriente", "mcaja_ahorro", "mcaja_ahorro_adicional", 
                "mcaja_ahorro_dolares", "mcuentas_saldo", "mautoservicio", "mtarjeta_visa_consumo", "mtarjeta_master_consumo", 
                "mprestamos_personales", "mprestamos_prendarios", "mprestamos_hipotecarios", "mplazo_fijo_dolares", 
                "mplazo_fijo_pesos", "minversion1_pesos", "minversion1_dolares", "minversion2", "mpayroll", "mpayroll2", 
                "mcuenta_debitos_automaticos", "mttarjeta_visa_debitos_automaticos", "mttarjeta_master_debitos_automaticos", 
                "mpagodeservicios", "mpagomiscuentas", "mcajeros_propios_descuentos", "mtarjeta_visa_descuentos", 
                "mtarjeta_master_descuentos", "mcomisiones_mantenimiento", "mcomisiones_otras", "mforex_buy", "mforex_sell", 
                "mtransferencias_recibidas", "mtransferencias_emitidas", "mextraccion_autoservicio", "mcheques_depositados", 
                "mcheques_emitidos", "mcheques_depositados_rechazados", "mcheques_emitidos_rechazados", "matm", "matm_other", 
                "Master_mfinanciacion_limite", "Master_msaldototal", "Master_msaldopesos", "Master_msaldodolares", 
                "Master_mconsumospesos", "Master_mconsumosdolares", "Master_mlimitecompra", "Master_madelantopesos", 
                "Master_madelantodolares", "Master_mpagado", "Master_mpagospesos", "Master_mpagosdolares", 
                "Master_mconsumototal", "Master_mpagominimo", "Visa_mfinanciacion_limite", "Visa_msaldototal", 
                "Visa_msaldopesos", "Visa_msaldodolares", "Visa_mconsumospesos", "Visa_mconsumosdolares", 
                "Visa_mlimitecompra", "Visa_madelantopesos", "Visa_madelantodolares", "Visa_mpagado", "Visa_mpagospesos", 
                "Visa_mpagosdolares", "Visa_mconsumototal", "Visa_mpagominimo")
rank_summary <- create_compact_monthly_summary(dataset, original_cols)
fwrite(rank_summary, file = "rank_summary.csv", sep = ",")
print(rank_summary)

# Create lag and delta features
exclude_fields <- c("numero_de_cliente", "foto_mes", "clase_ternaria")
fields_to_transform <- setdiff(names(dataset), exclude_fields)

cat("Creating lag and delta features...\n")
cat("  Fields to transform:", length(fields_to_transform), "\n")

setorder(dataset, numero_de_cliente, foto_mes)

# Lag features
dataset[, paste0(fields_to_transform, "_lag1") := 
  lapply(.SD, shift, n = 1), 
  by = numero_de_cliente, 
  .SDcols = fields_to_transform]

cat("  Lag features created\n")

# Delta features
dataset[, paste0(fields_to_transform, "_delta1") := 
  lapply(fields_to_transform, function(f) get(f) - get(paste0(f, "_lag1")))]

cat("  Delta features created\n")
cat("  Total columns now:", ncol(dataset), "\n")

# Create relacion_dependencia feature
dataset[, relacion_dependencia := FALSE]

# Identify and update cases that meet the condition
dataset[foto_mes == 202105 & 
     (mpayroll_lag1 / mpayroll) >= 1.3 & 
     (mpayroll_lag1 / mpayroll) <= 1.8, 
   relacion_dependencia := TRUE]

cat("La columa relacion dependencia tiene")
print(dataset[, .N, list(relacion_dependencia)])

# Target Variable Creation (clase_ternaria)
dsimple <- dataset[, list(
    "pos" = .I,
    numero_de_cliente,
    periodo0 = as.integer(foto_mes/100)*12 +  foto_mes%%100 ) ]

setorder( dsimple, numero_de_cliente, periodo0 )

periodo_ultimo <- dsimple[, max(periodo0) ]
periodo_anteultimo <- periodo_ultimo - 1

dsimple[, c("periodo1", "periodo2") :=
    shift(periodo0, n=1:2, fill=NA, type="lead"),  numero_de_cliente ]

dsimple[ periodo0 < periodo_anteultimo, clase_ternaria := "CONTINUA" ]

dsimple[ periodo0 < periodo_ultimo &
    ( is.na(periodo1) | periodo0 + 1 < periodo1 ),
    clase_ternaria := "BAJA+1" ]

dsimple[ periodo0 < periodo_anteultimo & (periodo0+1 == periodo1 )
    & ( is.na(periodo2) | periodo0 + 2 < periodo2 ),
    clase_ternaria := "BAJA+2" ]

setorder( dsimple, pos )
dataset[, clase_ternaria := dsimple$clase_ternaria ]

fwrite( dataset,
    file =  "/Users/manumoreira/Repos/dmeyf2025/Competencia1/data/competencia_01.8.csv.gz",
    sep = ","
)

setorder( dataset, foto_mes, clase_ternaria, numero_de_cliente)
print(dataset[, .N, list(foto_mes, clase_ternaria)])

# Helper Functions

# Stratified partitioning
particionar <- function(data, division, agrupa = "", campo = "fold", start = 1, seed = NA) {
  if (!is.na(seed)) set.seed(seed, "L'Ecuyer-CMRG")
  
  bloque <- unlist(mapply(
    function(x, y) rep(y, x), 
    division, 
    seq(from = start, length.out = length(division))
  ))
  
  data[, (campo) := sample(rep(bloque, ceiling(.N / length(bloque))))[1:.N], by = agrupa]
}

# Initialize reality dataset for gain evaluation
realidad_inicializar <- function(pfuture, pparam) {
  drealidad <- pfuture[, list(numero_de_cliente, foto_mes, clase_ternaria)]
  particionar(drealidad, division = c(3, 7), agrupa = "clase_ternaria", 
              seed = PARAM$semilla_kaggle)
  return(drealidad)
}

# Evaluate gain (simulates Kaggle split)
realidad_evaluar <- function(prealidad, pprediccion) {
  prealidad[pprediccion, on = c("numero_de_cliente", "foto_mes"), 
            predicted := i.Predicted]
  
  tbl <- prealidad[, list(qty = .N), by = list(fold, predicted, clase_ternaria)]
  
  res <- list()
  res$public <- tbl[fold == 1 & predicted == 1L, 
    sum(qty * ifelse(clase_ternaria == "BAJA+2", 780000, -20000))] / 0.3
  res$private <- tbl[fold == 2 & predicted == 1L, 
    sum(qty * ifelse(clase_ternaria == "BAJA+2", 780000, -20000))] / 0.7
  res$total <- tbl[predicted == 1L, 
    sum(qty * ifelse(clase_ternaria == "BAJA+2", 780000, -20000))]
  
  prealidad[, predicted := NULL]
  return(res)
}

# Objective function for Bayesian Optimization
EstimarGanancia_AUC_lightgbm <- function(x) {
  param_completo <- modifyList(PARAM$lgbm$param_fijos, x)
  
  modelocv <- lgb.cv(
    data = dtrain,
    nfold = PARAM$hyperparametertuning$xval_folds,
    stratified = TRUE,
    param = param_completo
  )
  
  AUC <- modelocv$best_score
  rm(modelocv)
  gc(full = TRUE, verbose = FALSE)
  
  message(format(Sys.time(), "%a %b %d %X %Y"), " AUC ", AUC)
  return(AUC)
}

cat("Helper functions defined\n")

# Create experiment directory
exp_name <- sprintf("exp%03d_seed_%d", PARAM$experimento, PARAM$semilla_primigenia)
exp_dir <- file.path(BASE_PATH, "results", exp_name)
dir.create(exp_dir, recursive = TRUE, showWarnings = FALSE)
setwd(exp_dir)

cat("Working directory:", getwd(), "\n")

# Filter training data
dataset_train <- dataset[foto_mes %in% PARAM$train]

cat("Training dataset:\n")
cat("  Total rows:", nrow(dataset_train), "\n")
print(dataset_train[, .N, by = clase_ternaria])

# Apply undersampling
set.seed(PARAM$semilla_primigenia, kind = "L'Ecuyer-CMRG")
dataset_train[, azar := runif(nrow(dataset_train))]
dataset_train[, training := 0L]

dataset_train[
  foto_mes %in% PARAM$train &
    (azar <= PARAM$trainingstrategy$undersampling | 
     clase_ternaria %in% c("BAJA+1", "BAJA+2")),
  training := 1L
]

cat("\nAfter undersampling:\n")
print(dataset_train[training == 1L, .N, by = clase_ternaria])
cat("  Total training rows:", dataset_train[training == 1L, .N], "\n")

# Clase01 binaria de la clase ternaria
dataset_train[,
  clase01 := ifelse(clase_ternaria %in% c("BAJA+2","BAJA+1"), 1L, 0L)
]

# Define features
campos_buenos <- setdiff(
  colnames(dataset_train),
  c("clase_ternaria", "clase01", "azar", "training")
)

cat("Features for training:", length(campos_buenos), "\n")

# Create LightGBM dataset
dtrain <- lgb.Dataset(
  data = data.matrix(dataset_train[training == 1L, campos_buenos, with = FALSE]),
  label = dataset_train[training == 1L, clase01],
  free_raw_data = FALSE
)

cat("LightGBM training dataset created:\n")
cat("  Rows:", nrow(dtrain), "\n")
cat("  Columns:", ncol(dtrain), "\n")

# Configure Bayesian Optimization
kbayesiana <- "bayesiana.RDATA"

configureMlr(show.learner.output = FALSE)

obj.fun <- makeSingleObjectiveFunction(
  fn = EstimarGanancia_AUC_lightgbm,
  minimize = FALSE,
  noisy = TRUE,
  par.set = PARAM$hypeparametertuning$hs,
  has.simple.signature = FALSE
)

ctrl <- makeMBOControl(
  save.on.disk.at.time = 600,
  save.file.path = kbayesiana
)

ctrl <- setMBOControlTermination(ctrl, iters = PARAM$hyperparametertuning$iteraciones)
ctrl <- setMBOControlInfill(ctrl, crit = makeMBOInfillCritEI())

surr.km <- makeLearner(
  "regr.km",
  predict.type = "se",
  covtype = "matern3_2",
  control = list(trace = TRUE)
)

cat("Bayesian Optimization configured\n")
cat("  Iterations:", PARAM$hyperparametertuning$iteraciones, "\n")

# Run Bayesian Optimization (this will take a while)
cat("\nStarting Bayesian Optimization...\n")
cat("This may take several hours depending on iterations\n\n")

if (!file.exists(kbayesiana)) {
  bayesiana_salida <- mbo(obj.fun, learner = surr.km, control = ctrl)
} else {
  cat("Continuing from existing bayesiana.RDATA\n")
  bayesiana_salida <- mboContinue(kbayesiana)
}

cat("\nBayesian Optimization completed\n")

# Save and analyze results
tb_bayesiana <- as.data.table(bayesiana_salida$opt.path)
tb_bayesiana[, iter := .I]
setorder(tb_bayesiana, -y)

fwrite(tb_bayesiana, file = "BO_log.txt", sep = "\t")

# Extract best hyperparameters
PARAM$out$lgbm$mejores_hiperparametros <- tb_bayesiana[1, 
  setdiff(colnames(tb_bayesiana),
    c("y", "dob", "eol", "error.message", "exec.time", "ei", "error.model",
      "train.time", "prop.type", "propose.time", "se", "mean", "iter")),
  with = FALSE
]

PARAM$out$lgbm$y <- tb_bayesiana[1, y]

cat("\nBest hyperparameters found:\n")
print(PARAM$out$lgbm$mejores_hiperparametros)
cat("\nBest AUC:", PARAM$out$lgbm$y, "\n")

write_yaml(PARAM, file = "PARAM.yml")
cat("\nParameters saved to PARAM.yml\n")

# Prepare final training dataset (includes validation period)
dataset_train_validate <- dataset[foto_mes %in% PARAM$train_validate]

dataset_train_validate[,
  clase01 := ifelse(clase_ternaria %in% c("BAJA+2","BAJA+1"), 1L, 0L)
]

cat("Final training dataset:\n")
cat("  Total rows:", nrow(dataset_train_validate), "\n")
print(dataset_train_validate[, .N, by = clase_ternaria])

dtrain_validate <- lgb.Dataset(
  data = data.matrix(dataset_train_validate[, campos_buenos, with = FALSE]),
  label = dataset_train_validate[, clase01]
)

cat("\nLightGBM dataset created for validation training\n")

# Prepare final parameters
param_final <- modifyList(PARAM$lgbm$param_fijos, 
                          PARAM$out$lgbm$mejores_hiperparametros)

# Normalize min_data_in_leaf for full dataset (no undersampling)
param_normalizado <- copy(param_final)
param_normalizado$min_data_in_leaf <- round(
  param_final$min_data_in_leaf / PARAM$trainingstrategy$undersampling
)

cat("Final parameters prepared\n")
cat("  Original min_data_in_leaf:", param_final$min_data_in_leaf, "\n")
cat("  Normalized min_data_in_leaf:", param_normalizado$min_data_in_leaf, "\n")

# Train final model
cat("\nTraining validate model...\n")
modelo_final <- lgb.train(data = dtrain_validate, param = param_normalizado)
cat("Final model trained successfully\n")

# Feature importance
tb_importancia <- as.data.table(lgb.importance(modelo_final))
fwrite(tb_importancia, file = "impo.txt", sep = "\t")

cat("\nTop 10 most important features:\n")
print(head(tb_importancia, 10))

# Save model
lgb.save(modelo_final, "modelo.txt")
cat("\nModel and importance saved\n")

# Predict on validation set
dvalidate <- dataset[foto_mes %in% PARAM$validate]

cat("Validation dataset:\n")
cat("  Rows:", nrow(dvalidate), "\n")
print(dvalidate[, .N, by = clase_ternaria])

prediccion <- predict(modelo_final, 
  data.matrix(dvalidate[, campos_buenos, with = FALSE]))

cat("\nPredictions generated\n")
cat("  Mean probability:", mean(prediccion), "\n")
cat("  Min probability:", min(prediccion), "\n")
cat("  Max probability:", max(prediccion), "\n")

# Prepare prediction table
tb_prediccion <- dvalidate[, list(numero_de_cliente, foto_mes)]
tb_prediccion[, prob := prediccion]

fwrite(tb_prediccion, file = "prediccion.txt", sep = "\t")

# Initialize reality dataset for gain evaluation
drealidad <- realidad_inicializar(dvalidate, PARAM)

cat("Prediction table created and saved\n")

# Generate submissions for different cutoffs
setorder(tb_prediccion, -prob)
dir.create("kaggle", showWarnings = FALSE)

gain_results <- data.table()

cat("\n--- Validation Gains ---\n")
cat(sprintf("%-10s %-15s %-15s %-15s %-15s\n", 
            "Envios", "Threshold", "Total", "Public", "Private"))
cat(strrep("-", 70), "\n")

for (envios in PARAM$cortes) {
  tb_prediccion[, Predicted := 0L]
  tb_prediccion[1:envios, Predicted := 1L]
  
  archivo_kaggle <- paste0("./kaggle/KA", PARAM$experimento, "_", envios, ".csv")
  fwrite(tb_prediccion[, list(numero_de_cliente, Predicted)],
    file = archivo_kaggle, sep = ",")
  
  res <- realidad_evaluar(drealidad, tb_prediccion)
  prob_threshold <- tb_prediccion[envios, prob]
  
  gain_results <- rbind(gain_results, data.table(
    envios = envios,
    prob_threshold = prob_threshold,
    gain_total = res$total,
    gain_public = res$public,
    gain_private = res$private
  ))
  
  cat(sprintf("%-10d %-15.6f %-15.0f %-15.0f %-15.0f\n",
    envios, prob_threshold, res$total, res$public, res$private))
}

# Save results
fwrite(gain_results, file = "gain_results.csv")
cat("\nGain results saved to gain_results.csv\n")

# Find optimal cutoff
optimal_row <- gain_results[which.max(gain_total)]

cat("\n=== OPTIMAL CUTOFF ===\n")
cat("Envios:", optimal_row$envios, "\n")
cat("Probability threshold:", optimal_row$prob_threshold, "\n")
cat("Total gain:", optimal_row$gain_total, "\n")
cat("Public gain:", optimal_row$gain_public, "\n")
cat("Private gain:", optimal_row$gain_private, "\n")

# Create plots
p1 <- ggplot(gain_results, aes(x = envios, y = gain_total)) +
  geom_line(size = 1) +
  geom_point() +
  geom_text(aes(x = optimal_row$envios, y = max(gain_total), 
                label = paste("Optimal:", optimal_row$envios)), 
            vjust = -1) +
  labs(title = "Total Gain vs Number of Submissions",
       x = "Number of Submissions",
       y = "Total Gain") +
  theme_minimal()

p2 <- ggplot(gain_results, aes(x = prob_threshold, y = gain_total)) +
  geom_line(size = 1) +
  geom_point() +
  geom_text(aes(x = optimal_row$prob_threshold, y = max(gain_total), 
                label = paste("Optimal:", round(optimal_row$prob_threshold, 4))), 
            vjust = -1) +
  labs(title = "Total Gain vs Probability Threshold",
       x = "Probability Threshold",
       y = "Total Gain") +
  theme_minimal()

ggsave("gain_vs_envios.png", p1, width = 8, height = 6)
ggsave("gain_vs_threshold.png", p2, width = 8, height = 6)

cat("\nPlots saved:\n")
cat("  - gain_vs_envios.png\n")
cat("  - gain_vs_threshold.png\n")