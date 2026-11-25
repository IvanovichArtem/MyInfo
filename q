# Отток оцениваем как снижение остатка к предыдущей отчётной дате.
change = balance.diff()
outflow = (-change).clip(lower=0)

weekday_stats = (
    pd.DataFrame({"balance": balance, "change": change, "outflow": outflow})
    .groupby(balance.index.dayofweek)
    .agg(**{
        "Дат": ("balance", "count"),
        "Медиана остатка, млн": ("balance", "median"),
        "Средний остаток, млн": ("balance", "mean"),
        "Среднее изменение, млн": ("change", "mean"),
        "Средний отток, млн": ("outflow", "mean"),
        "Суммарный отток, млн": ("outflow", lambda s: s.sum(min_count=1)),
        "Доля дат с оттоком, %": ("outflow", lambda s: s.dropna().gt(0).mean() * 100),
    })
    .reindex(range(7))
    .set_axis(["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"])
    .rename_axis("День недели")
)

display(weekday_stats.round(2))
