"""Тестовые данные формата настройки сценарного расчёта блендинга.

Это не допущения проекта: в config/settings.yaml компонентов смешения, присадок и остальных параметров смешения нет,
пока не заданы паспортные данные. Тесты подставляют эти значения в конфигурацию на время проверки.
"""
K, G = "Керосин гидроочищенный", "Газойль лёгкий гидроочищенный"
MAIN = "ГО ДТ (24-2000)"
TEST_BLENDING = {
    "share_step": {"value": 0.05}, "batch_t": {"value": 1000},
    "rule_margins": {"value": {"T95": 3.0, "CFPP": 2.0, "cetane": 1.0}},
    "main_stock_t": {"value": 100000}, "main_share_range": {"value": [0.0, 1.0]},
    "main_defaults": {"value": {"D15": 840.0, "cetane": 53.0, "CFPP": -8.0}},
    "manual_shares": {"value": {MAIN: 0.9, K: 0.05, G: 0.05}},
    "manual_doses": {"value": {"Присадка A (цетаноповышающая)": 0.0, "Присадка B (депрессорная)": 0.0}},
    "scenario_components": {
        K: {"enabled": True, "S": 3.0, "D15": 800.0, "cetane": 45.0, "T95": 255.0, "CFPP": -45.0, "stock_t": 250,
            "share_range": [0.0, 1.0]},
        G: {"enabled": True, "S": 9.0, "D15": 862.0, "cetane": 47.0, "T95": 352.0, "CFPP": -3.0, "stock_t": 400,
            "share_range": [0.0, 1.0]}},
    "additives": {
        "Присадка A (цетаноповышающая)": {"enabled": True, "param": "cetane", "doses_kg_t": [0.0, 0.5, 1.0], "effect_per_kg_t": 2.5},
        "Присадка B (депрессорная)": {"enabled": True, "param": "CFPP", "doses_kg_t": [0.0, 0.2, 0.4], "effect_per_kg_t": -15.0}},
}
