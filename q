def predict_and_evaluate_alternative(df: pd.DataFrame, id_col, dt_col, col, migration_matrix, view=True):
    df = df.copy()
    df = df.sort_values([id_col, dt_col]).reset_index(drop=True)
    migration_matrix = clean_migration_matrix(migration_matrix)
    state_list = list(migration_matrix.columns)

    def predict_from_initial(initial_state, steps):
        state_idx = state_list.index(initial_state)
        M_power = np.linalg.matrix_power(migration_matrix.values, steps)
        probs = M_power[state_idx, :]
        pred = state_list[np.argmax(probs)]
        if pred == "Погашено":
            return initial_state
        return pred

    predicted_values = []
    initial_states = df.groupby(id_col).first()[col].to_dict()

    for uid, group in df.groupby(id_col):
        group = group.sort_values(dt_col)
        init_state = initial_states[uid]
        preds = []
        for i, row in enumerate(group.itertuples(), start=0):
            if i == 0:
                preds.append(init_state)
            else:
                pred_state = predict_from_initial(init_state, i)
                preds.append(pred_state)
        predicted_values.extend(preds)

    df[f'predicted_next_{col}'] = predicted_values
    target = df[col].shift(-1).dropna()
    predict = df[f"predicted_next_{col}"].iloc[:-1]
    accuracy = accuracy_score(target, predict)
    print(f"\nAccuracy score: {accuracy:.4f}")
    labels = sorted(df[col].dropna().unique())
    conf_mat = confusion_matrix(target, predict, labels=labels)
    conf_df = pd.DataFrame(conf_mat, index=labels, columns=labels)
    if view:
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_df, annot=True, fmt='d', cmap='Greens', linewidths=0.5)
        plt.xlabel("Предсказано")
        plt.ylabel("Фактическое")
        plt.title("Confusion matrix (альтернатива)")
        plt.show()
    class_report = classification_report(target, predict, labels=labels, zero_division=0, output_dict=not view)
    return class_report
