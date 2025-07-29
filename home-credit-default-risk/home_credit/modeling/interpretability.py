import matplotlib.pyplot as plt
import numpy as np
import shap


def plot_feature_importance(model, feature_names, max_num_features=20, ax=None):
    """
    Affiche l'importance des variables pour les modèles à importance intégrée (arbres, etc.)
    """
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:max_num_features]
        if ax is None:
            fig, ax = plt.subplots()
        ax.barh(np.array(feature_names)[indices][::-1], importances[indices][::-1])
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importances")
        plt.tight_layout()
        plt.show()
    else:
        raise ValueError("Le modèle ne possède pas d'attribut feature_importances_.")


def plot_shap_global(model, X, max_display=20):
    """
    Affiche le summary plot SHAP (interprétabilité globale)
    """
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap.summary_plot(shap_values, X, max_display=max_display)


def plot_shap_local(model, X, instance_idx=0):
    """
    Affiche l'explication SHAP pour une instance donnée (interprétabilité locale)
    """
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap.plots.waterfall(shap_values[instance_idx]) 