from ftip.alpha_kernel import StructuralAlphaKernel
from ftip.superfactor import SuperfactorModel
from ftip.features import FeatureEngineer


def test_structural_alpha_kernel(sample_data):
    features = FeatureEngineer().build_feature_matrix(sample_data)
    kernel = StructuralAlphaKernel(n_factors=2)
    kernel.fit(features)
    alpha = kernel.structural_alpha(features)
    assert alpha.name == "structural_alpha"
    assert len(alpha) == len(sample_data)


def test_superfactor_model(sample_data):
    features = FeatureEngineer().build_feature_matrix(sample_data)
    model = SuperfactorModel()
    superfactor = model.fit_transform(features)
    assert superfactor.name == "superfactor_alpha"
    assert len(superfactor) == len(sample_data)
