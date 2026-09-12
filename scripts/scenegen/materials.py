"""Deterministic preview materials and material binding helpers."""

from pxr import Gf, Sdf, UsdGeom, UsdShade

from .params import *  # noqa: F401,F403


# =========================================================
# MATERIAL / VISIBILITY HELPERS
# =========================================================

def create_preview_material(
    stage,
    path,
    color,
    roughness=0.65,
    metallic=0.0,
):
    """
    Create one reusable USD Preview Surface material.
    """
    material = UsdShade.Material.Define(
        stage,
        path,
    )

    shader = UsdShade.Shader.Define(
        stage,
        f"{path}/PreviewSurface",
    )

    shader.CreateIdAttr(
        "UsdPreviewSurface"
    )

    shader.CreateInput(
        "diffuseColor",
        Sdf.ValueTypeNames.Color3f,
    ).Set(
        Gf.Vec3f(
            float(color[0]),
            float(color[1]),
            float(color[2]),
        )
    )

    shader.CreateInput(
        "roughness",
        Sdf.ValueTypeNames.Float,
    ).Set(
        float(roughness)
    )

    shader.CreateInput(
        "metallic",
        Sdf.ValueTypeNames.Float,
    ).Set(
        float(metallic)
    )

    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(),
        "surface",
    )

    return material


def create_scene_materials(
    stage,
):
    """
    Create all deterministic perception-baseline materials once.
    """
    stage.DefinePrim(
        "/World/Looks",
        "Scope",
    )

    materials = {
        "ground": create_preview_material(
            stage,
            "/World/Looks/Ground",
            SCENE_COLORS["ground"],
            roughness=0.90,
        ),
        "bed": create_preview_material(
            stage,
            "/World/Looks/SoilBed",
            SCENE_COLORS["bed"],
            roughness=0.95,
        ),
        "main_stem": create_preview_material(
            stage,
            "/World/Looks/MainStem",
            SCENE_COLORS["main_stem"],
            roughness=0.75,
        ),
        "peduncle": create_preview_material(
            stage,
            "/World/Looks/Peduncle",
            SCENE_COLORS["peduncle"],
            roughness=0.75,
        ),
        "branch": create_preview_material(
            stage,
            "/World/Looks/TomatoBranch",
            SCENE_COLORS["branch"],
            roughness=0.75,
        ),
        "leaf": create_preview_material(
            stage,
            "/World/Looks/Leaf",
            SCENE_COLORS["leaf"],
            roughness=0.82,
        ),
        "tomato": create_preview_material(
            stage,
            "/World/Looks/Tomato",
            SCENE_COLORS["tomato"],
            roughness=0.55,
        ),
        "grasp_marker": create_preview_material(
            stage,
            "/World/Looks/GraspMarker",
            SCENE_COLORS["grasp_marker"],
            roughness=0.35,
        ),
        "cut_marker": create_preview_material(
            stage,
            "/World/Looks/CutMarker",
            SCENE_COLORS["cut_marker"],
            roughness=0.35,
        ),
    }

    return materials


def bind_scene_material(
    prim,
    material_name,
):
    """
    Bind one of SCENE_MATERIALS to a prim.
    """
    material = SCENE_MATERIALS.get(
        material_name
    )

    if material is None:
        raise RuntimeError(
            f"Scene material not initialized: {material_name}"
        )

    UsdShade.MaterialBindingAPI.Apply(
        prim
    ).Bind(
        material
    )


def bind_scene_material_recursive(
    prim,
    material_name,
):
    """
    Bind material to every renderable Gprim under a prim.
    Useful for GroundPlane, whose visible geometry may be nested.
    """
    if not prim.IsValid():
        return

    if prim.IsA(
        UsdGeom.Gprim
    ):
        bind_scene_material(
            prim,
            material_name,
        )

    for child in prim.GetChildren():
        bind_scene_material_recursive(
            child,
            material_name,
        )


def set_render_visibility(
    prim,
    visible,
):
    """
    Keep prim in USD but control whether it appears in rendered RGB.
    """
    imageable = UsdGeom.Imageable(
        prim
    )

    if not imageable:
        return

    imageable.CreateVisibilityAttr().Set(
        (
            UsdGeom.Tokens.inherited
            if visible
            else UsdGeom.Tokens.invisible
        )
    )
