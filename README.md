# Atmospheric New Particle Formation (NPF) Classifier
With this project, we build and assess classifiers that forecast day-to-day NPF event types (class4 ∈ {Ia, Ib, II, nonevent})
as well as the resulting binary indicator for the event (class2 = event vs nonevent); while assessing
classifiers using three metrics - class2 accuracy, perplexity (based on predicted binary probabilities for
event / nonevent), multiclass accuracy (class4). An overall score will be issued on all three metrics
combined. Our pipeline focuses on robust probability estimates (to minimize perplexity) while
maintaining strong multiclass accuracy by using feature engineering, three base models that have been
well regularized (Extra Trees, XGBoost, Random Forest), tuning of hyperparameters using Optuna, a
search for the optimal ensemble weight for out-of-fold cross-validation predictions, and calibration of the
binary probabilities through Platt Scaling. We have integrated MLflow for rigorous logging of parameters, metrics, and model artifacts, ensuring a fully reproducible research pipeline.
