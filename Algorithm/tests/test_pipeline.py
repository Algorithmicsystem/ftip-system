from ftip.pipeline import FatTailPipeline

def test_pipeline_run(sample_data):
    pipeline = FatTailPipeline(horizon=1, threshold=0)
    output = pipeline.run(sample_data)
    assert "weight" in output.columns
    assert "label" in output.columns
    assert output["weight"].sum() > 0
