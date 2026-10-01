from model import train_model

def test_model_accuracy():
    _, accuracy = train_model()
    assert accuracy >= 0.90
