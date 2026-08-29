from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class SupportMode(StrEnum):
    EXISTING_10 = "EXISTING_10"
    LOWER_TRUSS_ENDS = "LOWER_TRUSS_ENDS"
    CUSTOM = "CUSTOM"


class BaseFixity(StrEnum):
    FIXED = "FIXED"
    PINNED = "PINNED"


class JointModel(StrEnum):
    PINNED = "PINNED"
    FRAME = "FRAME"


class TrussType(StrEnum):
    RECTANGULAR = "RECTANGULAR"
    TRIANGULAR = "TRIANGULAR"
    TRAPEZOIDAL = "TRAPEZOIDAL"


class WebType(StrEnum):
    WARREN = "WARREN"
    PRATT = "PRATT"
    HOWE = "HOWE"
    CROSS = "CROSS"


class StructureMode(StrEnum):
    TWO_LEVEL = "TWO_LEVEL"
    ONE_LEVEL = "ONE_LEVEL"


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


@dataclass(frozen=True)
class Point3D:
    x: float
    y: float
    z: float


@dataclass
class CanopyInput:
    preset_name: str = "original"
    width: float = 12.0
    left_depth: float = 10.0
    right_depth: float = 7.0
    lower_count: int = 4
    lower_positions_x: list[float] | None = field(default_factory=lambda: [0.0, 4.0, 8.0, 12.0])
    upper_count: int = 11
    lower_height_mm: float = 250.0
    upper_height_mm: float = 250.0
    lower_chord_section: str = "60x40x3"
    lower_web_section: str = "40x40x2"
    upper_chord_section: str = "60x40x3"
    upper_web_section: str = "40x40x2"
    column_section: str = "80x80x3"
    support_mode: SupportMode = SupportMode.EXISTING_10
    allow_interior_columns: bool = False
    base_fixity: BaseFixity = BaseFixity.FIXED
    column_joint: BaseFixity = BaseFixity.PINNED
    truss_joint_model: JointModel = JointModel.PINNED
    truss_type: TrussType = TrussType.RECTANGULAR
    web_type: WebType = WebType.WARREN
    lower_web_type: WebType = WebType.WARREN
    upper_web_type: WebType = WebType.WARREN
    panel_count: int | None = None
    lower_target_panel_length: float = 1.0
    target_panel_length: float = 1.0
    front_left_height: float = 3.0
    front_right_height: float = 3.0
    rear_left_height: float = 3.0
    roof_dead_load_kpa: float = 0.15
    snow_sg_kpa: float = 1.90
    snow_ce: float = 1.0
    snow_ct: float = 1.0
    snow_mu: float | None = None
    snow_gamma_f: float = 1.4
    steel_grade: str = "C245"
    steel_resistance_mpa: float = 245.0
    steel_price_per_kg: float = 95.0
    fabrication_per_truss: float = 0.0
    column_cost: float = 0.0
    foundation_cost: float = 0.0
    connection_cost: float = 0.0
    structure_mode: StructureMode = StructureMode.TWO_LEVEL

    def __post_init__(self) -> None:
        self.support_mode = SupportMode(self.support_mode)
        self.base_fixity = BaseFixity(self.base_fixity)
        self.column_joint = BaseFixity(self.column_joint)
        self.truss_joint_model = JointModel(self.truss_joint_model)
        self.truss_type = TrussType(self.truss_type)
        self.web_type = WebType(self.web_type)
        self.lower_web_type = WebType(self.lower_web_type)
        self.upper_web_type = WebType(self.upper_web_type)
        self.structure_mode = StructureMode(self.structure_mode)


@dataclass
class Node:
    id: int
    x: float
    y: float
    z: float
    label: str
    roles: set[str] = field(default_factory=set)

    @property
    def coords(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class Element:
    id: int
    i: int
    j: int
    role: str
    group: str
    section: str
    label: str


@dataclass
class Column:
    name: str
    x: float
    y: float
    base_node: int
    top_node: int


@dataclass
class StructuralModel:
    nodes: dict[int, Node]
    elements: list[Element]
    columns: list[Column]
    lower_trusses: dict[str, list[int]]
    upper_trusses: dict[str, list[int]]
    upper_bottom_trusses: dict[str, list[int]]
    roof_node_ids: list[int]
    support_node_ids: list[int]
    loaded_roof_area: float
    warnings: list[str] = field(default_factory=list)


@dataclass
class LoadBreakdown:
    sg_kpa: float
    ce: float
    ct: float
    mu: float
    gamma_f: float
    s0_kpa: float
    sd_kpa: float
    dead_load_kpa: float
    total_roof_load_n: float
    total_self_weight_n: float
