from __future__ import annotations

from config.models import ClassificationConfig


def _build_example_json(cls_config: ClassificationConfig) -> str:
    import json as _json

    cats = cls_config.categories
    tones = cls_config.tones

    first_cat = cats[0].id if cats else ""
    second_cat = cats[1].id if len(cats) > 1 else first_cat
    last_cat = cats[-1].id if cats else ""
    first_tone = tones[0].id if tones else ""
    last_tone = tones[-1].id if len(tones) > 1 else first_tone

    relevant_full: dict = {
        "mentions_disability": True,
        "categories": [first_cat, second_cat],
        "tones": [first_tone],
    }

    relevant_partial: dict = {
        "mentions_disability": True,
        "categories": [last_cat],
        "tones": [last_tone],
    }

    irrelevant: dict = {
        "mentions_disability": False,
        "categories": [],
        "tones": [],
    }

    if cls_config.problem_scopes:
        relevant_full["problem_scope"] = cls_config.problem_scopes[0].id
        relevant_partial["problem_scope"] = ""
        irrelevant["problem_scope"] = ""
    if cls_config.disability_models:
        relevant_full["disability_model"] = cls_config.disability_models[0].id
        relevant_partial["disability_model"] = ""
        irrelevant["disability_model"] = ""
    if cls_config.agency_types:
        relevant_full["agency"] = cls_config.agency_types[0].id
        last_agency = (
            cls_config.agency_types[-1].id
            if len(cls_config.agency_types) > 1
            else ""
        )
        relevant_partial["agency"] = last_agency
        irrelevant["agency"] = ""
    if cls_config.portrayal_types:
        relevant_full["portrayal"] = cls_config.portrayal_types[0].id
        relevant_partial["portrayal"] = ""
        irrelevant["portrayal"] = ""
    if cls_config.target_audiences:
        relevant_full["target_audiences"] = [cls_config.target_audiences[0].id]
        relevant_partial["target_audiences"] = []
        irrelevant["target_audiences"] = []
    if cls_config.barriers:
        relevant_full["barriers"] = [cls_config.barriers[0].id]
        relevant_partial["barriers"] = []
        irrelevant["barriers"] = []
    if cls_config.stigma_types:
        relevant_full["stigma"] = cls_config.stigma_types[0].id
        relevant_partial["stigma"] = ""
        irrelevant["stigma"] = ""

    return _json.dumps(
        {"results": [relevant_full, relevant_partial, irrelevant]},
        ensure_ascii=False,
        indent=2,
    )


def build_system_prompt(cls_config: ClassificationConfig) -> str:
    categories_desc = "\n".join(
        f"- {c.id}: {c.label}" + (f" -- {c.description}" if c.description else "")
        for c in cls_config.categories
    )
    tones_desc = "\n".join(
        f"- {t.id}: {t.label}" for t in cls_config.tones
    )
    example_json = _build_example_json(cls_config)

    extra_sections = ""
    extra_fields = ""
    field_num = 3

    if cls_config.problem_scopes:
        scopes_desc = "\n".join(
            f"- {s.id}: {s.label}" + (f" -- {s.description}" if s.description else "")
            for s in cls_config.problem_scopes
        )
        extra_sections += (
            f"\n\n<problem_scope instruction=\"выбери один\">\n"
            f"Масштаб проблемы, поднятой в сообщении:\n{scopes_desc}\n"
            f"</problem_scope>"
        )
        field_num += 1
        extra_fields += (
            f"\n{field_num}. problem_scope (str): ID масштаба проблемы. "
            'Пустая строка "" если не определяется из текста.'
        )

    if cls_config.disability_models:
        models_desc = "\n".join(
            f"- {d.id}: {d.label}" + (f" -- {d.description}" if d.description else "")
            for d in cls_config.disability_models
        )
        extra_sections += (
            f"\n\n<disability_model instruction=\"выбери одну\">\n"
            f"Концептуальная рамка, через которую представлена инвалидность:\n"
            f"{models_desc}\n</disability_model>"
        )
        field_num += 1
        extra_fields += (
            f"\n{field_num}. disability_model (str): ID модели инвалидности. "
            'Пустая строка "" если не определяется из текста.'
        )

    if cls_config.agency_types:
        agency_desc = "\n".join(
            f"- {a.id}: {a.label}" + (f" -- {a.description}" if a.description else "")
            for a in cls_config.agency_types
        )
        extra_sections += (
            f"\n\n<agency instruction=\"выбери одну\">\n"
            f"Как представлен человек с инвалидностью в контексте собственной "
            f"субъектности:\n{agency_desc}\n</agency>"
        )
        field_num += 1
        extra_fields += (
            f"\n{field_num}. agency (str): ID типа агентности. "
            'Пустая строка "" если не определяется из текста.'
        )

    if cls_config.portrayal_types:
        portrayal_desc = "\n".join(
            f"- {p.id}: {p.label}" + (f" -- {p.description}" if p.description else "")
            for p in cls_config.portrayal_types
        )
        extra_sections += (
            f"\n\n<portrayal instruction=\"выбери одну\">\n"
            f"Как изображён человек с инвалидностью в сообщении:\n"
            f"{portrayal_desc}\n</portrayal>"
        )
        field_num += 1
        extra_fields += (
            f"\n{field_num}. portrayal (str): ID типа репрезентации. "
            'Пустая строка "" если не определяется из текста.'
        )

    if cls_config.target_audiences:
        ta_desc = "\n".join(
            f"- {t.id}: {t.label}" + (f" -- {t.description}" if t.description else "")
            for t in cls_config.target_audiences
        )
        extra_sections += (
            f"\n\n<target_audiences instruction=\"назначь ВСЕ подходящие\">\n"
            f"Кому адресовано сообщение:\n{ta_desc}\n</target_audiences>"
        )
        field_num += 1
        extra_fields += (
            f"\n{field_num}. target_audiences (list[str]): ID подходящих адресатов. "
            "Пустой список если не определяется из текста."
        )

    if cls_config.barriers:
        barriers_desc = "\n".join(
            f"- {b.id}: {b.label}" + (f" -- {b.description}" if b.description else "")
            for b in cls_config.barriers
        )
        extra_sections += (
            f"\n\n<barriers instruction=\"назначь ВСЕ подходящие\">\n"
            f"Какие барьеры упоминаются в сообщении:\n{barriers_desc}\n</barriers>"
        )
        field_num += 1
        extra_fields += (
            f"\n{field_num}. barriers (list[str]): ID упоминаемых барьеров. "
            "Пустой список если барьеры не упоминаются в тексте."
        )

    if cls_config.stigma_types:
        stigma_desc = "\n".join(
            f"- {s.id}: {s.label}" + (f" -- {s.description}" if s.description else "")
            for s in cls_config.stigma_types
        )
        extra_sections += (
            f"\n\n<stigma instruction=\"выбери одну\">\n"
            f"Наличие и тип стигмы в сообщении:\n{stigma_desc}\n</stigma>"
        )
        field_num += 1
        extra_fields += (
            f"\n{field_num}. stigma (str): ID типа стигмы. "
            'Пустая строка "" если стигма не проявляется в тексте.'
        )

    return f"""\
<task>
Ты -- исследовательский ассистент. Твоя единственная задача -- КЛАССИФИЦИРОВАТЬ \
каждое сообщение из Telegram-каналов по заданной схеме.
</task>

<rules>
- Ты НЕ суммаризируешь и НЕ пересказываешь сообщения. Ты КЛАССИФИЦИРУЕШЬ.
- Каждое сообщение обрабатывай НЕЗАВИСИМО от остальных.
- Выбирай ID значений СТРОГО из предложенных списков.
- Классифицируй ТОЛЬКО на основе того, что ЯВНО написано в тексте. \
НЕ додумывай контекст и НЕ делай предположений.
</rules>

<dimensions>

<categories instruction="назначь ВСЕ подходящие">
{categories_desc}
</categories>

<tones instruction="назначь ВСЕ подходящие">
{tones_desc}
</tones>{extra_sections}

</dimensions>

<response_format>
Поля для КАЖДОГО сообщения:

1. mentions_disability (bool): упоминается ли инвалидность, люди с инвалидностью \
или смежные темы? ЭТО ГЛАВНОЕ РЕШЕНИЕ.
2. categories (list[str]): ID подходящих категорий. Пустой список если ни одна \
не подходит.
3. tones (list[str]): ID подходящих тональностей. Пустой список если ни одна \
не подходит.{extra_fields}
</response_format>

<algorithm>
ШАГ 1: Определи mentions_disability.
- Если false -> ВСЕ остальные поля пустые ([], ""). Перейди к следующему сообщению.

ШАГ 2 (только если mentions_disability=true):
- Назначь categories и tones строго по содержанию текста.
- Для остальных полей: заполняй ТОЛЬКО если в тексте есть ЯВНОЕ основание. \
Если основания нет -- ставь "" или [].
- При сомнении между двумя категориями -- назначь обе.
- НЕ заполняй поле только потому что оно есть в схеме -- пустое значение допустимо.
</algorithm>

<example description="3 сообщения: полная классификация, частичная, нерелевантное">
{example_json}
</example>

Ты получишь нумерованный список сообщений. Верни JSON с массивом "results" -- \
по одному объекту на сообщение, в том же порядке."""


def build_user_prompt(messages: list[tuple[int, str]]) -> str:
    """Build the user prompt with numbered messages.

    messages: list of (index, text) tuples.
    """
    lines = [f"Сообщений: {len(messages)}.\n"]
    for idx, text in messages:
        truncated = text[:2000] if len(text) > 2000 else text
        lines.append(f"[Сообщение {idx}]\n{truncated}\n")
    return "\n".join(lines)
