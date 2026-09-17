"""词典采样和模型标注共用的标签体系；不依赖具体标注实现。"""

TAXONOMY_VERSION = "rust-aspects-v2"

ASPECTS = frozenset(
    {
        "ownership",
        "type_system",
        "safety",
        "runtime_performance",
        "learning_curve",
        "compile_time",
        "diagnostics_debugging",
        "tooling_documentation",
        "readability_maintainability",
        "api_extensibility",
        "package_manager",
        "libraries_frameworks",
        "community",
    }
)
CLASSES = frozenset({"negative", "neutral", "positive"})
