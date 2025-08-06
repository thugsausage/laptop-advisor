import csv
import random
import os  # Добавили модуль для работы с файловой системой
from faker import Faker

fake = Faker()


def generate_laptops(num_products=20):
    brands = ["Lenovo", "Dell", "HP", "Asus", "Acer", "Apple", "MSI", "Samsung"]
    cpus = ["Intel i5", "Intel i7", "Intel i9", "AMD Ryzen 5", "AMD Ryzen 7", "AMD Ryzen 9", "Apple M1", "Apple M2"]
    models = {
        "Lenovo": ["ThinkPad X1", "ThinkBook", "IdeaPad", "Legion"],
        "Dell": ["XPS 13", "XPS 15", "Inspiron", "Alienware"],
        "HP": ["Spectre", "Envy", "Pavilion", "Omen"],
        "Asus": ["ZenBook", "ROG Zephyrus", "VivoBook", "TUF Gaming"],
        "Acer": ["Swift", "Aspire", "Predator", "Nitro"],
        "Apple": ["MacBook Air", "MacBook Pro"],
        "MSI": ["Stealth", "Raider", "Katana", "Sword"],
        "Samsung": ["Galaxy Book", "Odyssey"]
    }

    products = []
    for _ in range(num_products):
        brand = random.choice(brands)
        model_base = random.choice(models[brand])
        model_suffix = fake.bothify(text="?##") if brand != "Apple" else ""
        model = f"{model_base} {model_suffix}".strip()

        variants = []
        for _ in range(random.randint(1, 3)):
            ram = random.choice([8, 16, 32, 64])
            cpu = random.choice([c for c in cpus if not ("Apple" in c and brand != "Apple")])
            base_price = 500 + (ram / 8 * 200)
            if "i7" in cpu or "Ryzen 7" in cpu: base_price += 300
            if "i9" in cpu or "Ryzen 9" in cpu: base_price += 600
            if "Apple" in cpu: base_price += 400
            price = round(base_price * (0.9 + random.random() * 0.3), 2)

            variants.append({
                "id": fake.uuid4(),
                "brand": brand,
                "model": model,
                "ram_gb": ram,
                "cpu": cpu,
                "price": price,
                "in_stock": random.choice([True, False])
            })

        products.extend(variants)
    return products


def save_to_csv(data, filename="data/laptops.csv"):
    # Создаем папку, если она не существует
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)


if __name__ == "__main__":
    laptops = generate_laptops()
    save_to_csv(laptops)
    print(f"Сгенерировано {len(laptops)} вариантов ноутбуков. Файл сохранен в data/laptops.csv")
    print("Пример записи:")
    print(laptops[0])