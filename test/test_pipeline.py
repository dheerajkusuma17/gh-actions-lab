from src.evaluate_model import compute_metrics, passes_gate
from src.train_model import build_model, load_data, split_data

def test_data_has_expected_shape():
    X,y = load_data()
    assert X.shape == (569, 30)
    assert set(y) == {0, 1}

def test_split_keeps_20_percent_for_testing():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    assert len(X_test) == 114
    assert len(X_train) + len(X_test) == len(X)

def test_model_learns_well():
    X,y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = build_model(n_estimators=50, learning_rate=0.1, max_depth=3)
    model.fit(X_train,y_train)
    metrics = compute_metrics(model, X_test, y_test)
    assert metrics["f1_score"] > 0.90

def test_quality_gate_blocks_weak_models():
    assert passes_gate({"f1_score": 0.50}, min_f1=0.95) is False
    assert passes_gate({"f1_score": 0.97}, min_f1=0.95) is True