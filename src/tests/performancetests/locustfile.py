from locust import HttpUser, task, between
import random


class TextDetectionUser(HttpUser):
    # Wait between 1 and 5 seconds between tasks
    wait_time = between(1, 5)

    # Sample texts for testing with varying lengths and content
    sample_texts = [
        "This is a short human-written text for testing.",
        "A longer piece of text that could have been written by either a human or AI. It contains multiple sentences and tries to mimic real-world usage patterns that your API might encounter in production.",
        "Dear valued customer, Thank you for your recent purchase. We appreciate your business and hope you are satisfied with our product. Please don't hesitate to contact us if you need any assistance.",
        "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the English alphabet at least once.",
        """In a groundbreaking study published today, researchers discovered a novel approach to sustainable energy production.
        The findings suggest that through a combination of advanced materials and innovative design, efficiency rates could be
        improved by up to 40% compared to current methods.""",
    ]

    def on_start(self):
        """Initialize the user session."""
        # Make a health check request when the user starts
        self.client.get("/health")

    @task(1)
    def check_health(self):
        """Task to check API health."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                if response.json()["status"] == "healthy":
                    response.success()
                else:
                    response.failure("Health check indicated unhealthy status")
            else:
                response.failure(f"Health check failed with status code: {response.status_code}")

    @task(1)
    def get_root(self):
        """Task to test the root endpoint."""
        with self.client.get("/", catch_response=True) as response:
            if response.status_code == 200:
                if "message" in response.json():
                    response.success()
                else:
                    response.failure("Root response missing message field")
            else:
                response.failure(f"Root request failed with status code: {response.status_code}")

    @task(3)
    def predict_text(self):
        """Task to test the prediction endpoint with random sample texts."""
        # Randomly select a sample text
        text = random.choice(self.sample_texts)

        payload = {"text": text}
        headers = {"Content-Type": "application/json"}

        with self.client.post("/predict", json=payload, headers=headers, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if all(
                    key in data
                    for key in ["message", "predicted_class", "human_probability", "ai_probability", "raw_logits"]
                ):
                    response.success()
                else:
                    response.failure("Prediction response missing required fields")
            else:
                response.failure(f"Prediction failed with status code: {response.status_code}")

    @task(1)
    def predict_invalid_input(self):
        """Task to test the prediction endpoint with invalid input."""
        payload = {"text": ""}  # Empty text should trigger validation error
        headers = {"Content-Type": "application/json"}

        with self.client.post("/predict", json=payload, headers=headers, catch_response=True) as response:
            if response.status_code == 422:  # Expected validation error
                response.success()
            else:
                response.failure(f"Expected 422 status code, got {response.status_code}")


class WarmupUser(HttpUser):
    """A separate user class for initial warmup phase with lower wait times."""

    wait_time = between(0.1, 0.5)
    weight = 1

    @task
    def warmup(self):
        """Simple warmup task hitting the health endpoint."""
        self.client.get("/health")
