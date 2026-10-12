"""Research half-roof loads, integrated over tributary cells, not SP20 prescriptions."""
from src.loads.tributary import tributary_roof_loads

PATTERNS = {
    "UNIFORM": "Равномерный снег",
    "LEFT_HALF": "Снег на левой половине (X)",
    "RIGHT_HALF": "Снег на правой половине (X)",
    "FRONT_HALF": "Снег на передней половине (Y)",
    "REAR_HALF": "Снег на задней половине (Y)",
}


def snow_pattern_loads(data, model, pressure_kpa):
    if data.snow_pattern not in PATTERNS:
        raise ValueError("Неизвестная схема снега")
    if data.snow_pattern == "UNIFORM":
        return tributary_roof_loads(data, model, pressure_kpa)
    if data.preset_name != "naves_one":
        raise ValueError("Половинные загружения пока доступны только для NavesOne")
    from src.geometry.naves_one import UPPER_YS, LEFT_REAR_Y, WIDTH, _cell_area
    loads = {}
    for row, node_ids in enumerate(model.upper_trusses.values()):
        y = UPPER_YS[row]
        y0 = 0.0 if row == 0 else (UPPER_YS[row - 1] + y) / 2
        y1 = LEFT_REAR_Y if row == len(UPPER_YS) - 1 else (y + UPPER_YS[row + 1]) / 2
        xs = [model.nodes[n].x for n in node_ids]
        for col, node in enumerate(node_ids):
            x0 = -1.0 if col == 0 else (xs[col - 1] + xs[col]) / 2
            x1 = WIDTH + 1.0 if col == len(xs) - 1 else (xs[col] + xs[col + 1]) / 2
            low_x, high_x, low_y, high_y = x0, x1, y0, y1
            if data.snow_pattern == "LEFT_HALF":
                high_x = min(high_x, WIDTH / 2)
            elif data.snow_pattern == "RIGHT_HALF":
                low_x = max(low_x, WIDTH / 2)
            elif data.snow_pattern == "FRONT_HALF":
                high_y = min(high_y, LEFT_REAR_Y / 2)
            else:
                low_y = max(low_y, LEFT_REAR_Y / 2)
            area = _cell_area(low_x, high_x, low_y, high_y) if high_x > low_x and high_y > low_y else 0.0
            loads[node] = -1000 * pressure_kpa * area
    return loads
