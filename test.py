# from huggingface_hub import login
# login(token="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")

from pyannote.audio import Model
segmentation_model = Model.from_pretrained(
    "pyannote/segmentation", 
    use_auth_token="hf18_SbBzzjXRcJDzZOnxJzppqpTIUJYGSAjjoX"
)
