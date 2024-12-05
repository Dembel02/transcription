import requests

url = "https://huggingface.co/CRD716/ggml-vicuna-1.1-quantized/resolve/main/ggml-vicuna-13B-1.1-q4_0.bin"
# url = "https://huggingface.co/IlyaGusev/saiga_7b_lora_llamacpp/resolve/main/ggml-model-q4_1.bin"

output_path = "ggml-vicuna-13B-1.1-q4_0.bin"
# output_path = "ggml-model-q4_1.bin"

response = requests.get(url, stream=True)
if response.status_code == 200:
    with open(output_path, "wb") as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    print("Файл успешно загружен!")
else:
    print(f"Ошибка при загрузке: {response.status_code}")