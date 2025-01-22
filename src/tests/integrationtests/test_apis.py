from fastapi.testclient import TestClient
from text_detect.app import app


client = TestClient(app)


def test_read_root():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200

    data = response.json()

    assert "message" in data
    assert data["message"] == "Welcome to the Text Detection API!"
    assert "version" in data
    assert data["version"] == "1.0.0"
    assert "docs" in data
    assert data["docs"] == "/docs"
    assert "health_check" in data
    assert data["health_check"] == "/health"
    assert "predict_endpoint" in data
    assert data["predict_endpoint"] == "/predict"


def test_health_check():
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200

    data = response.json()

    assert "status" in data
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert isinstance(data["model_loaded"], bool)


def test_predict_valid_input():
    """Test the predict endpoint with valid input."""
    payload = {"text": "This is a sample text written by a human."}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert "message" in data
    assert "predicted_class" in data
    assert data["predicted_class"] in ["Human", "AI"]
    assert "human_probability" in data
    assert "ai_probability" in data
    assert "raw_logits" in data
    assert isinstance(data["raw_logits"], list)
    assert len(data["raw_logits"]) > 0


def test_predict_invalid_input():
    """Test the predict endpoint with invalid input (empty text)."""
    payload = {"text": ""}  # Invalid input: empty string
    response = client.post("/predict", json=payload)

    # Assert that the status code is 422 (Validation error)
    assert response.status_code == 422

    # Check the error message
    assert "detail" in response.json()
    assert response.json()["detail"][0]["msg"] == "String should have at least 1 character"
