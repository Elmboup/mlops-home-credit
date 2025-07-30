from home_credit.modeling.train import train_model

def test_model_training():
    model, accuracy = train_model()

    # Le modèle doit avoir une méthode predict
    assert hasattr(model, "predict")

    # Le score doit être raisonnable
    assert accuracy > 0.5, f"Accuracy trop faible : {accuracy}"
