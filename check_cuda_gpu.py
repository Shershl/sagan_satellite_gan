import torch
import time

print("🔍 Перевірка наявності CUDA:")

if torch.cuda.is_available():
    print("✅ CUDA доступна")
    print("🖥️ Використовується GPU:", torch.cuda.get_device_name(0))
    print("🚀 Виконуємо тестове обчислення на GPU...")

    device = torch.device("cuda")

    a = torch.randn((10000, 10000), device=device)
    b = torch.randn((10000, 10000), device=device)

    start = time.time()
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
    end = time.time()

    print(f"✅ Обчислення завершено на GPU за {end - start:.3f} с")

else:
    print("❌ CUDA не знайдена — тренування йде на CPU")
