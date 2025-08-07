import pytest
import os
import pandas as pd
from unittest.mock import patch, MagicMock
from advisor import LaptopAdvisor


@pytest.fixture(scope="module")
def advisor():
    return LaptopAdvisor("data/laptops.csv")


# Тест 1: Загрузка данных
def test_load_products(advisor):
    assert len(advisor.products) > 0, "Данные не загружены"
    assert isinstance(advisor.products[0], dict), "Неверный формат данных"


# Тест 2: Доступные бренды
def test_available_brands(advisor):
    brands_in_data = {p["brand"] for p in advisor.products}
    assert set(advisor.available_brands) == brands_in_data, "Списки брендов не совпадают"


# Тест 3: Точный поиск бренда
def test_exact_brand_match(advisor):
    test_brand = advisor.products[0]["brand"]
    result = advisor._fuzzy_match_brand(test_brand)
    assert result == test_brand, "Не найдено точное совпадение бренда"


# Тест 4: Поиск бренда с опечаткой
def test_fuzzy_brand_match(advisor):
    test_brand = advisor.products[0]["brand"]
    typo_brand = test_brand[:-1] + "x"
    result = advisor._fuzzy_match_brand(typo_brand)
    assert result == test_brand, "Не найдено совпадение при опечатке"


# Тест 5: Неудачный поиск бренда
def test_fuzzy_brand_no_match(advisor):
    result = advisor._fuzzy_match_brand("NonexistentBrand123")
    assert result is None, "Ошибочно найден несуществующий бренд"


# Тест 6: Извлечение фильтра RAM
@patch("advisor.OpenAI")
def test_extract_filters_ram(mock_openai, advisor):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"ram": 16}'))]
    )

    filters = advisor._extract_filters("ноутбук с 16GB RAM")
    assert filters.get("ram") == 16, "Неверно извлечен фильтр RAM"


# Тест 7: Извлечение фильтров через GPT
@patch("advisor.OpenAI")
def test_extract_filters_gpt(mock_openai, advisor):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client
    mock_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content='{"max_price": 1000, "cpu": "Intel i5"}'))]
    )

    filters = advisor._extract_filters("ноутбук до 1000 с i5")
    assert filters.get("max_price") == 1000, "Неверно извлечена максимальная цена"
    assert "i5" in filters.get("cpu", "").lower(), "Неверно извлечен процессор"


# Тест 8: Фильтр наличия товара
def test_extract_filters_in_stock(advisor):
    filters = advisor._extract_filters("ноутбуки в наличии")
    assert filters.get("in_stock") is True, "Неверно извлечен фильтр наличия"


# Тест 9: Применение фильтра по RAM
def test_apply_filters_ram(advisor):
    test_ram = min(p["ram_gb"] for p in advisor.products)
    filtered = advisor._apply_filters({"ram": test_ram})
    assert all(p["ram_gb"] >= test_ram for p in filtered), "Неверная фильтрация по RAM"


# Тест 10: Применение фильтра по бренду
def test_apply_filters_brand(advisor):
    test_brand = advisor.products[0]["brand"]
    filtered = advisor._apply_filters({"brand": test_brand})
    assert all(p["brand"] == test_brand for p in filtered), "Неверная фильтрация по бренду"


# Тест 11: Применение фильтра по цене
def test_apply_filters_max_price(advisor):
    test_price = max(p["price"] for p in advisor.products)
    filtered = advisor._apply_filters({"max_price": test_price})
    assert all(p["price"] <= test_price for p in filtered), "Неверная фильтрация по цене"


# Тест 12: Применение фильтра наличия
def test_apply_filters_in_stock(advisor):
    filtered = advisor._apply_filters({"in_stock": True})
    assert all(p["in_stock"] for p in filtered), "Неверная фильтрация по наличию"


# Тест 13: Учет предпочтений пользователя
def test_apply_filters_with_preference(advisor):
    test_brand = advisor.products[0]["brand"]
    advisor.preferences["brand"] = test_brand
    filtered = advisor._apply_filters({})
    assert all(p["brand"] == test_brand for p in filtered), "Не учитывается предпочтение по бренду"
    advisor.preferences["brand"] = None


# Тест 14: Форматирование результатов
def test_format_results(advisor):
    """Проверка форматирования результатов"""
    test_products = advisor.products[:2]
    result = advisor._format_results(test_products)
    assert f"Найдено ноутбуков: {len(test_products)}" in result, "Неверное количество в выводе"
    assert all(p["brand"] in result for p in test_products), "Не все бренды в выводе"


# Тест 15: Сравнение моделей
def test_compare_products(advisor):
    advisor.last_results = advisor.products[:3]
    result = advisor._compare_products([1, 2])
    assert "Сравнение" in result, "Нет заголовка сравнения"
    assert "Рекомендация" in result, "Нет рекомендации"


# Тест 16: Ошибка сравнения
def test_compare_products_invalid(advisor):
    advisor.last_results = advisor.products[:2]
    result = advisor._compare_products([5])
    assert "Неверные номера моделей" in result, "Нет сообщения об ошибке"


# Тест 17: Установка предпочтений
def test_set_preference(advisor):
    test_brand = advisor.products[0]["brand"]
    response = advisor.process_command(f"предпочитаю {test_brand}")
    assert f"Запомнил ваше предпочтение: {test_brand}" in response, "Нет подтверждения"
    assert advisor.preferences["brand"] == test_brand, "Предпочтение не сохранено"
    advisor.preferences["brand"] = None  # Сброс


# Тест 18: Ошибка установки предпочтений
def test_set_preference_invalid(advisor):
    response = advisor.process_command("предпочитаю InvalidBrand123")
    assert "не найден" in response, "Нет сообщения об ошибке"


# Тест 19: Команда выхода
def test_exit_command(advisor):
    response = advisor.process_command("выход")
    assert response == "exit", "Неверный результат команды выхода"


# Тест 20: Рекомендации (fallback)
def test_recommendation_fallback(advisor):
    original_client = advisor.client

    try:
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Error")
        advisor.client = mock_client

        result = advisor._get_full_recommendation(advisor.products[:2])

        assert "❌" in result, "Нет маркера ошибки"
        assert any(p["brand"] in result for p in advisor.products[:2]), "Нет списка ноутбуков"
    finally:
        advisor.client = original_client


# Тест 21: Отображение цен в долларах
@patch("advisor.OpenAI")
def test_currency_display(mock_openai, advisor):
    mock_client = MagicMock()
    mock_openai.return_value = mock_client

    # Мок ответа с евро
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content="Цена: 1000€"))]
    mock_client.chat.completions.create.return_value = mock_response

    result = advisor._get_full_recommendation(advisor.products[:1])
    assert "€" not in result, "Найдены цены в евро"
    assert "$" in result, "Не найдены цены в долларах"


if __name__ == "__main__":
    pytest.main(["-v", "test_advisor.py"])