import os
import sys
import json
import re
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv
import pandas as pd
from rapidfuzz import fuzz, process

# Настройка кодировки
try:
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stdin.reconfigure(encoding='utf-8')
except (AttributeError, OSError):
    pass

# Загрузка переменных окружения
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    print("Ошибка: Не найден HF_TOKEN в файле .env")
    exit(1)

class LaptopAdvisor:
    def __init__(self, data_path: str = "data/laptops.csv"):
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=HF_TOKEN,
        )
        self.products = self._load_products(data_path)
        self.last_results = []
        self.current_filters = {}
        self.preferences = {
            'brand': None,
            'min_ram': None,
            'max_price': None,
            'cpu': None,
            'in_stock': None
        }
        self.available_brands = list(set(p['brand'] for p in self.products))

    def _load_products(self, path: str) -> List[Dict]:
        """Загрузка данных о ноутбуках из CSV"""
        try:
            df = pd.read_csv(path)
            df['price'] = df['price'].astype(float)
            df['ram_gb'] = df['ram_gb'].astype(int)
            df['in_stock'] = df['in_stock'].astype(bool)
            return df.to_dict('records')
        except Exception as e:
            print(f"Ошибка загрузки данных: {e}")
            return []

    def _fuzzy_match_brand(self, query: str, threshold: int = 75) -> Optional[str]:
        """Поиск бренда с учетом опечаток"""
        if not query:
            return None
        query_lower = query.lower()
        for brand in self.available_brands:
            if brand.lower() == query_lower:
                return brand
        result = process.extractOne(query, self.available_brands, scorer=fuzz.WRatio)
        if result and result[1] >= threshold:
            return result[0]
        return None

    def _extract_filters(self, user_input: str) -> Dict:
        """Извлечение параметров фильтрации из запроса"""
        filters = {}

        if "в наличии" in user_input.lower():
            filters["in_stock"] = True

        ram_match = re.search(r'(\d+)\s*gb|рам\s*(\d+)', user_input.lower())
        if ram_match:
            ram_value = ram_match.group(1) or ram_match.group(2)
            filters["ram"] = int(ram_value)

        prompt = (
            "Извлеки параметры ноутбука из запроса на русском. "
            "Возможные параметры: ram (int), max_price (float), cpu (str), brand (str), in_stock (bool). "
            "Для in_stock используй true/false. "
            f"Запрос: {user_input} "
            "Верни только JSON, например: {\"cpu\": \"Intel i7\", \"in_stock\": true}"
        )

        try:
            response = self.client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            if response and response.choices:
                gpt_filters = json.loads(response.choices[0].message.content)
                for key, value in gpt_filters.items():
                    if key not in filters:
                        filters[key] = value
                if "brand" in filters:
                    matched_brand = self._fuzzy_match_brand(filters["brand"])
                    if matched_brand:
                        filters["brand"] = matched_brand
                    else:
                        del filters["brand"]
                return filters
            return filters
        except Exception as e:
            print(f"Ошибка при извлечении фильтров: {e}")
            return filters

    def _apply_filters(self, filters: Dict) -> List[Dict]:
        """Применение фильтров"""
        filtered = self.products
        self.current_filters.update(filters)

        if self.preferences['brand']:
            filtered = [p for p in filtered if p['brand'].lower() == self.preferences['brand'].lower()]

        if "ram" in self.current_filters:
            filtered = [p for p in filtered if p['ram_gb'] >= self.current_filters["ram"]]
        if "max_price" in self.current_filters:
            filtered = [p for p in filtered if p['price'] <= self.current_filters["max_price"]]
        if "cpu" in self.current_filters:
            filtered = [p for p in filtered if self.current_filters["cpu"].lower() in p['cpu'].lower()]
        if "brand" in self.current_filters:
            matched_brand = self._fuzzy_match_brand(self.current_filters["brand"])
            if matched_brand:
                filtered = [p for p in filtered if p['brand'].lower() == matched_brand.lower()]
        if "in_stock" in self.current_filters:
            filtered = [p for p in filtered if p['in_stock'] == self.current_filters["in_stock"]]

        return filtered

    def _get_full_recommendation(self, products: List[Dict]) -> str:
        """Получение полной рекомендации с исправлением валюты"""
        if not products:
            return "❌ Нет ноутбуков для рекомендации"

        prompt = (
            "Ты эксперт по выбору ноутбуков. Выбери лучший вариант из списка и обоснуй выбор на русском языке.\n"
            f"Предпочтения пользователя: {json.dumps(self.preferences, ensure_ascii=False)}\n"
            f"Текущие фильтры: {json.dumps(self.current_filters, ensure_ascii=False)}\n"
            "Ноутбуки для анализа:\n"
            f"{json.dumps(products[:10], ensure_ascii=False, indent=2)}\n"
            "ВАЖНО: Все цены должны быть указаны в ДОЛЛАРАХ ($), а не в евро!\n"
            "Учитывай:\n"
            "1. Соответствие требованиям\n"
            "2. Соотношение цены и качества\n"
            "3. Наличие в магазине\n"
            "4. Технические характеристики\n"
            "Формат ответа:\n"
            "🏆 Рекомендуемый ноутбук: [полное название]\n"
            "📌 Характеристики: [основные параметры]\n"
            "💡 Обоснование: [развернутое объяснение на русском языке]"
        )

        try:
            response = self.client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            if response and response.choices:
                recommendation = response.choices[0].message.content
                # Заменяем евро на доллары в рекомендации
                recommendation = recommendation.replace("€", "$")
                recommendation = re.sub(r'(\d+)\s*евро', r'$\1', recommendation, flags=re.IGNORECASE)
                return recommendation
        except Exception as e:
            print(f"Ошибка получения рекомендации: {e}")
            # Формируем fallback с информацией о продуктах
            fallback = ["❌ Не удалось получить рекомендацию. Доступные варианты:"]
            fallback.extend(f"- {p['brand']} {p['model']} (${p['price']:.2f})" for p in products[:3])
        return "\n".join(fallback)

    def _format_product(self, product: Dict) -> str:
        """Форматирование информации о ноутбуке"""
        return (
            f"{product['brand']} {product['model']} | "
            f"RAM: {product['ram_gb']}GB | "
            f"CPU: {product['cpu']} | "
            f"Цена: ${product['price']:.2f} | "
            f"{'✅ В наличии' if product['in_stock'] else '❌ Нет в наличии'}"
        )

    def _format_results(self, products: List[Dict]) -> str:
        """Форматирование списка ноутбуков"""
        if not products:
            return "❌ Не найдено ноутбуков по заданным критериям."
        result = f"🔍 Найдено ноутбуков: {len(products)}\n"
        for i, p in enumerate(products[:10], 1):
            result += f"{i}. {self._format_product(p)}\n"
        if len(products) > 10:
            result += f"\nПоказано 10 из {len(products)}. Уточните критерии."
        return result

    def _compare_products(self, indices: List[int]) -> str:
        """Сравнение выбранных моделей"""
        if not self.last_results:
            return "❌ Нет результатов для сравнения. Сначала выполните поиск."

        selected = []
        for idx in indices:
            if 1 <= idx <= len(self.last_results):
                selected.append(self.last_results[idx - 1])

        if not selected:
            return f"❌ Неверные номера моделей. Доступны номера от 1 до {len(self.last_results)}."

        simplified = [
            {
                'brand': p['brand'],
                'model': p['model'],
                'cpu': p['cpu'],
                'ram_gb': p['ram_gb'],
                'price': f"${p['price']:.2f}",  # Явно указываем доллары
                'in_stock': p['in_stock']
            }
            for p in selected
        ]

        prompt = (
            "Сравни следующие ноутбуки на русском языке и дай рекомендацию:\n"
            f"{json.dumps(simplified, ensure_ascii=False, indent=2)}\n"
            "ВАЖНО: Все цены должны быть указаны в ДОЛЛАРАХ ($), а не в евро!\n"
            "Сделай:\n"
            "1. Детальное сравнение характеристик\n"
            "2. Оценку по соотношению цена/качество\n"
            "3. Рекомендацию лучшего варианта с обоснованием\n"
            "Формат вывода на русском:\n"
            "📊 Сравнение ноутбуков:\n"
            "[подробное сравнение в табличной или списковой форме]\n"
            "🏆 Рекомендация:\n"
            "[развернутое обоснование выбора]"
        )

        try:
            response = self.client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2
            )
            if response and response.choices:
                comparison = response.choices[0].message.content
                # Заменяем евро на доллары в сравнении
                comparison = comparison.replace("€", "$")
                comparison = re.sub(r'(\d+)\s*евро', r'$\1', comparison, flags=re.IGNORECASE)
                return comparison
        except Exception as e:
            print(f"Ошибка сравнения: {e}")

        # Fallback сравнение
        comparison = "📊 Сравнение:\n"
        for i, p in enumerate(selected, 1):
            comparison += (
                f"\n{i}. {p['brand']} {p['model']}\n"
                f"   CPU: {p['cpu']}, RAM: {p['ram_gb']}GB\n"
                f"   Цена: ${p['price']:.2f}\n"
                f"   {'✅ В наличии' if p['in_stock'] else '❌ Нет в наличии'}"
            )

        best = min(selected, key=lambda x: (0 if x['in_stock'] else 1, x['price'] / x['ram_gb']))
        comparison += (
            f"\n\n🏆 Рекомендация: {best['brand']} {best['model']}\n"
            f"💡 Почему: Лучшее соотношение цены и характеристик среди выбранных."
        )
        return comparison

    def process_command(self, user_input: str) -> str:
        """Обработка команд пользователя"""
        user_input_lower = user_input.strip().lower()
        original_input = user_input.strip()

        if user_input_lower in ["выход", "exit", "quit"]:
            return "exit"

        if user_input_lower.startswith("предпочитаю"):
            brand = original_input.split(maxsplit=1)[1].strip()
            matched_brand = self._fuzzy_match_brand(brand)
            if matched_brand:
                self.preferences['brand'] = matched_brand
                return f"✅ Запомнил ваше предпочтение: {matched_brand}"
            return f"❌ Бренд '{brand}' не найден. Доступные бренды: {', '.join(self.available_brands)}"

        if user_input_lower.startswith(("сравни", "compare")):
            numbers = [int(match) for match in re.findall(r'\d+', original_input)]
            return self._compare_products(numbers)

        new_filters = self._extract_filters(original_input)
        filtered = self._apply_filters(new_filters)
        self.last_results = filtered.copy()

        if any(word in user_input_lower for word in ["рекоменд", "совет", "посоветуй"]):
            return self._get_full_recommendation(self.last_results)

        return self._format_results(self.last_results)


def main():
    advisor = LaptopAdvisor()
    print("💻 Привет! Я помогу выбрать ноутбук. Задайте параметры или 'выход'")
    print("Доступные команды:")
    print("- 'покажи ноутбуки с Intel i7' - поиск по параметрам")
    print("- 'рекомендуй' - получить рекомендацию")
    print("- 'предпочитаю БРЕНД' - установить предпочтение по бренду")
    print("- 'сравни 1 2 3' - сравнить выбранные модели")
    print("- 'выход' - завершить работу\n")

    while True:
        try:
            user_input = input("Вы: ")
            response = advisor.process_command(user_input)
            if response == "exit":
                print("👋 До свидания!")
                break
            print("\n" + response + "\n")
        except KeyboardInterrupt:
            print("\n👋 До свидания!")
            break
        except Exception as e:
            print(f"\n⚠️ Ошибка: {str(e)}\n")


if __name__ == "__main__":
    main()