import kfp
from kfp import dsl
from kfp.dsl import component, Output, Input, Dataset, Model, Metrics


@component(
    base_image="python:3.9",
    packages_to_install=["scikit-learn==1.6.1", "pandas==2.0.3", "numpy==1.24.3"]
)
def load_data(dataset: Output[Dataset]):
    from sklearn.datasets import load_iris
    import pandas as pd

    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['target'] = iris.target

    df.to_csv(dataset.path, index=False)
    print(f"Data saved: {df.shape[0]} rows, {df.shape[1]} cols")
    print(df.head())

@component(
    base_image="python:3.9",
    packages_to_install=["scikit-learn==1.6.1", "pandas==2.0.3", "numpy==1.24.3"]
)
def train_model(dataset: Input[Dataset], model: Output[Model]):
    import pandas as pd
    import pickle
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(dataset.path)

    X = df.drop('target', axis=1)
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    with open(model.path, 'wb') as f:
        pickle.dump({'model': clf, 'X_test': X_test, 'y_test': y_test}, f)


@component(
    base_image="python:3.9",
    packages_to_install=["scikit-learn==1.6.1", "pandas==2.0.3", "numpy==1.24.3"]
)
def evaluate_model(model: Input[Model], metrics: Output[Metrics]):
    import pickle
    from sklearn.metrics import accuracy_score, classification_report

    with open(model.path, 'rb') as f:
        data = pickle.load(f)

    clf = data['model']
    X_test = data['X_test']
    y_test = data['y_test']

    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(classification_report(y_test, predictions))

    metrics.log_metric("accuracy", float(accuracy))
    metrics.log_metric("model_type", "RandomForest")
    metrics.log_metric("n_estimators", 100)


@dsl.pipeline(
    name="iris-mlops-pipeline"
)
def iris_pipeline():
    load_task = load_data()
    train_task = train_model(dataset=load_task.outputs['dataset'])
    evaluate_task = evaluate_model(model=train_task.outputs['model'])


if __name__ == "__main__":
    kfp.compiler.Compiler().compile(
        pipeline_func=iris_pipeline,
        package_path="iris_pipeline.yaml"
    )
    print("Pipeline compiled: iris_pipeline.yaml")
