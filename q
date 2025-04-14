def predict_and_evaluate(df: pd.DataFrame, id_col, dt_col, col, migration_matrix, p=False, view=True):
    df = df.copy()
    df = df.sort_values([id_col, dt_col]).reset_index(drop=True)
    predictions = []
    skip_ids = set()
    groups = df.groupby(id_col, sort=False)

    for gid, gdf in groups:
        current_state = gdf[col].iloc[0]
        for idx, row in gdf.iterrows():
            if gid in skip_ids:
                predictions.append((idx, "Списания"))
                continue
            if p:
                if current_state in migration_matrix.index:
                    next_st = np.random.choice(
                        migration_matrix.columns,
                        p=migration_matrix.loc[current_state].values
                    )
                    if next_st == "Погашено":
                        next_st = current_state
                    elif next_st == "Списания":
                        skip_ids.add(gid)
                else:
                    next_st = current_state
            else:
                if current_state in migration_matrix.index:
                    next_st = migration_matrix.loc[current_state].idxmax()
                else:
                    next_st = current_state
            predictions.append((idx, next_st))
            current_state = next_st

    predictions.sort(key=lambda x: x[0])
    pred_vals = [p[1] for p in predictions]
    df[f'predicted_next_{col}'] = pred_vals

    target = df[col].shift(-1).dropna()
    predict = df[f'predicted_next_{col}'].iloc[:-1]

    acc = accuracy_score(target, predict)
    print(f"\nAccuracy score {acc:.4f}")

    labels = sorted(df[col].dropna().unique())
    cm = confusion_matrix(target, predict, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    if view:
        print("\nConfusion matrix:")
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', linewidths=0.5)
        plt.xlabel("Предсказано")
        plt.ylabel("Фактическое")
        plt.title("Confusion matrix")
        plt.show()

    cr = classification_report(target, predict, labels=labels, zero_division=0, output_dict=not view)
    return cr
