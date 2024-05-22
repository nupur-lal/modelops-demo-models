from teradataml import (
    DataFrame,
    GLM,
    ScaleFit,
    ScaleTransform
)
from aoa import (
    record_training_stats,
    save_plot,
    aoa_create_context,
    ModelContext
)
import numpy as np


def plot_feature_importance(fi, img_filename):
    import pandas as pd
    import matplotlib.pyplot as plt
    feat_importances = pd.Series(fi)
    feat_importances.nlargest(10).plot(kind='barh').set_title('Feature Importance')
    fig = plt.gcf()
    fig.savefig(img_filename, dpi=500)
    plt.clf()


def train(context: ModelContext, **kwargs):
    aoa_create_context()

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]
    entity_key = context.dataset_info.entity_key

    # read training dataset from Teradata and convert to pandas
    train_df = DataFrame.from_query(context.dataset_info.sql)

  
    print("Starting training...")

    model = GLM(
        input_columns = feature_names,
        response_column = target_name,
        data = train_df,
        family = context.hyperparams["family"],
        learning_rate = context.hyperparams["learning_rate"],
        momentum = context.hyperparams["momentum"],
        initial_eta = context.hyperparams["initial_eta"],
        local_sgd_iterations = context.hyperparams["local_sgd_iterations"],
        iter_max = context.hyperparams["iter_max"],
        batch_size = context.hyperparams["batch_size"],
        iter_num_no_change = context.hyperparams["iter_num_no_change"]
    )
    
    model.result.to_sql(f"model_${context.model_version}", if_exists="replace")    
    print("Saved trained model")

    # Calculate feature importance and generate plot
    model_pdf = model.result.to_pandas()[['predictor','estimate']]
    predictor_dict = {}
    
    for index, row in model_pdf.iterrows():
        if row['predictor'] in feature_names:
            value = row['estimate']
            predictor_dict[row['predictor']] = value
    
    feature_importance = dict(sorted(predictor_dict.items(), key=lambda x: x[1], reverse=True))
    keys, values = zip(*feature_importance.items())
    norm_values = (values-np.min(values))/(np.max(values)-np.min(values))
    feature_importance = {keys[i]: float(norm_values[i]*1000) for i in range(len(keys))}
    plot_feature_importance(feature_importance, f"{context.artifact_output_path}/feature_importance")

    record_training_stats(
        train_df,
        features=feature_names,
        targets=[target_name],
        categorical=[ 'Churn','InternetService','Contract','PaymentMethod',
                'Gender_0','Gender_1','Partner_0','Partner_1','SeniorCitizen',
                  'Dependents_0','Dependents_1','PhoneService_0','PhoneService_1',
                  'MultipleLines_0','MultipleLines_1','OnlineSecurity_0','OnlineSecurity_1',
                  'OnlineBackup_0','OnlineBackup_1','DeviceProtection_0','DeviceProtection_1',
                  'TechSupport_0','TechSupport_1','StreamingTV_0','StreamingTV_1',
                  'StreamingMovies_0','StreamingMovies_1','PaperlessBilling_0',
                  'PaperlessBilling_1'],
        feature_importance=feature_importance,
        context=context
    )
