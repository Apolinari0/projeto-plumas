"""Convenções do vento (ERA5) e comparação com o deslocamento na imagem.

ERA5: u10 = componente para LESTE, v10 = componente para NORTE (m/s).
Direção meteorológica = de ONDE o vento sopra, em graus no sentido horário a
partir do Norte (vento de oeste = 270°).
"""
from __future__ import annotations

import math


def wind_speed(u: float, v: float) -> float:
    return math.hypot(u, v)


def wind_from_deg(u: float, v: float) -> float:
    """Direção meteorológica (de onde o vento vem), 0-360°."""
    return math.degrees(math.atan2(-u, -v)) % 360.0


def wind_to_deg(u: float, v: float) -> float:
    """Direção PARA onde o vento vai, 0-360°."""
    return (wind_from_deg(u, v) + 180.0) % 360.0


def lateral_wind_component(u: float, v: float, camera_azimuth_deg: float) -> float:
    """Componente do vento perpendicular à linha de visada da câmera.

    Positivo = vento empurra para a DIREITA da imagem; negativo = para a esquerda.
    A componente ao longo da visada (aproximando/afastando da câmera) não é
    observável como deslocamento lateral e fica de fora. Aproximação que ignora
    perspectiva e variação do vento com a altura.

    camera_azimuth_deg: direção para onde a câmera aponta (0=N, 90=L, horário).
    """
    az = math.radians(camera_azimuth_deg)
    # direita da câmera = (cos az, -sin az) em (leste, norte)
    return u * math.cos(az) - v * math.sin(az)


def sign_agreement(dx_px: float, lateral_ms: float, dead_zone_px: float = 1.0,
                   dead_zone_ms: float = 0.3) -> str:
    """'concorda', 'discorda' ou 'indefinido' (movimento/vento lateral pequeno).

    Compatibilidade de sinais é evidência de associação possível, não prova
    de causalidade.
    """
    if abs(dx_px) < dead_zone_px or abs(lateral_ms) < dead_zone_ms:
        return "indefinido"
    return "concorda" if (dx_px > 0) == (lateral_ms > 0) else "discorda"
